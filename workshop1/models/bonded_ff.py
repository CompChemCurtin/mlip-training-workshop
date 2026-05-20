"""Bonded force-field: harmonic bonds + harmonic angles + Fourier dihedrals.

This is the next step up from `pair_morse`: instead of summing over every
pair within `r_max`, we explicitly enumerate bonds, angles, and dihedrals
from a *fixed topology* and learn classical-FF-shaped parameters for them.

Topology is inferred once, from the first training frame, using
covalent-radius-scaled distance cutoffs. This works for rMD17 ethanol
because every frame is the same molecule. For mixtures or bond-breaking
data this assumption breaks, which is itself a teaching point about
classical force fields.

Functional forms (the usual MM/AMBER-style ones):

    E_bond     = k_b (r - r0)^2
    E_angle    = k_a (theta - theta0)^2
    E_dihedral = sum_{n=1,2,3} V_n [1 + cos(n*phi - delta_n)]
    E_total    = sum_bonds + sum_angles + sum_dihedrals + sum_i E0[Z_i] + E_offset

The bonded terms are all non-negative and the per-element E0 baselines fix
the dissociation limit, so without `E_offset` the model could never reach
the (much lower) energy of the actual molecule. Classical force fields
hide this same scalar inside their per-atom-type parameter set; we make
it explicit.

Parameter sharing: each instance is assigned a *type* by the unordered
tuple of element numbers along its path. There are typically a handful of
types per dataset (ethanol: 4 bond types, 5 angle types, 4 dihedral types).
All instances of a type share parameters, mirroring how classical FFs are
parameterised per atom-type tuple.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from workshop1.frames import Frame


# Covalent radii in Angstroms (Cordero et al., 2008). We use these to
# decide which pairs of atoms count as bonded in the initial topology.
_COV_RADII: Dict[int, float] = {
    1: 0.31,   # H
    6: 0.76,   # C
    7: 0.71,   # N
    8: 0.66,   # O
    9: 0.57,   # F
    15: 1.07,  # P
    16: 1.05,  # S
    17: 1.02,  # Cl
}


# ---------- topology inference ------------------------------------------

def _bond_cutoff(z1: int, z2: int, scale: float = 1.2) -> float:
    r1 = _COV_RADII.get(int(z1))
    r2 = _COV_RADII.get(int(z2))
    if r1 is None or r2 is None:
        raise ValueError(f"missing covalent radius for Z={z1 if r1 is None else z2}")
    return scale * (r1 + r2)


def infer_topology(frame: Frame, bond_scale: float = 1.2) -> Tuple[List[Tuple[int, int]],
                                                                   List[Tuple[int, int, int]],
                                                                   List[Tuple[int, int, int, int]]]:
    """Return (bonds, angles, dihedrals) as lists of atom-index tuples."""
    Z = frame.atomic_numbers.tolist()
    pos = frame.positions.detach().cpu().numpy()
    N = len(Z)

    # build adjacency
    neighbours: List[List[int]] = [[] for _ in range(N)]
    bonds: List[Tuple[int, int]] = []
    for i in range(N):
        for j in range(i + 1, N):
            d = float(((pos[i] - pos[j]) ** 2).sum() ** 0.5)
            if d < _bond_cutoff(Z[i], Z[j], scale=bond_scale):
                bonds.append((i, j))
                neighbours[i].append(j)
                neighbours[j].append(i)

    # angles: every (i, j, k) where j is bonded to both i and k, i < k
    angles: List[Tuple[int, int, int]] = []
    for j in range(N):
        nb = sorted(neighbours[j])
        for a in range(len(nb)):
            for b in range(a + 1, len(nb)):
                angles.append((nb[a], j, nb[b]))

    # dihedrals: every (i, j, k, l) along a 4-atom path
    dihedrals: List[Tuple[int, int, int, int]] = []
    for (j, k) in bonds:
        for i in neighbours[j]:
            if i == k:
                continue
            for l in neighbours[k]:
                if l == j or l == i:
                    continue
                # canonical orientation: smaller endpoint first
                tup = (i, j, k, l) if i < l else (l, k, j, i)
                dihedrals.append(tup)
    # de-duplicate (each dihedral can appear twice via the two bond directions)
    dihedrals = sorted(set(dihedrals))
    bonds.sort()
    angles.sort()
    return bonds, angles, dihedrals


# ---------- type bucketing ----------------------------------------------

def _bond_type(zi: int, zj: int) -> Tuple[int, int]:
    return tuple(sorted((int(zi), int(zj))))


def _angle_type(zi: int, zj: int, zk: int) -> Tuple[int, Tuple[int, int]]:
    # central atom kept fixed; flank-atoms order-insensitive
    return (int(zj), tuple(sorted((int(zi), int(zk)))))


def _dihedral_type(zi: int, zj: int, zk: int, zl: int) -> Tuple[int, int, int, int]:
    # palindromic: same type if read in reverse
    fwd = (int(zi), int(zj), int(zk), int(zl))
    rev = fwd[::-1]
    return min(fwd, rev)


def _assign_types(instances: Sequence[tuple], atomic_numbers: torch.Tensor,
                  type_fn) -> Tuple[List[int], List[tuple]]:
    """Return (type-index-per-instance, unique-type-list)."""
    z = atomic_numbers.tolist()
    type_keys = [type_fn(*(z[i] for i in inst)) for inst in instances]
    unique = sorted(set(type_keys))
    lookup = {k: i for i, k in enumerate(unique)}
    return [lookup[k] for k in type_keys], unique


# ---------- the model ---------------------------------------------------

class BondedFF(nn.Module):
    """Harmonic bonds + harmonic angles + 3-term Fourier dihedrals.

    Constructed from a single reference frame whose topology defines the
    bond/angle/dihedral indices and types. All frames passed at predict()
    time are assumed to share that topology (atom order preserved).
    """

    def __init__(self, ref_frame: Frame, atomic_energies: Dict[int, float],
                 bond_scale: float = 1.2, offset_init: float = 0.0):
        super().__init__()
        bonds, angles, dihedrals = infer_topology(ref_frame, bond_scale=bond_scale)
        Z = ref_frame.atomic_numbers

        # bonds
        b_types, b_unique = _assign_types(bonds, Z, _bond_type)
        self.register_buffer("bond_idx",  torch.as_tensor(bonds, dtype=torch.long))
        self.register_buffer("bond_type", torch.as_tensor(b_types, dtype=torch.long))
        self.bond_unique = b_unique
        # harmonic bond: k_b (r - r0)^2 ; both per-type, k_b kept >= 0 via softplus
        self.bond_log_k = nn.Parameter(torch.full((len(b_unique),), 1.0))
        self.bond_r0    = nn.Parameter(torch.full((len(b_unique),), 1.4))

        # angles
        a_types, a_unique = _assign_types(angles, Z, _angle_type)
        self.register_buffer("angle_idx",  torch.as_tensor(angles, dtype=torch.long))
        self.register_buffer("angle_type", torch.as_tensor(a_types, dtype=torch.long))
        self.angle_unique = a_unique
        self.angle_log_k  = nn.Parameter(torch.full((len(a_unique),), 0.0))
        self.angle_theta0 = nn.Parameter(torch.full((len(a_unique),), 1.91))   # ~109.5 deg

        # dihedrals
        if dihedrals:
            d_types, d_unique = _assign_types(dihedrals, Z, _dihedral_type)
        else:
            d_types, d_unique = [], []
        self.register_buffer("dihedral_idx",  torch.as_tensor(dihedrals or [(0,0,0,0)], dtype=torch.long))
        self.register_buffer("dihedral_type", torch.as_tensor(d_types or [0], dtype=torch.long))
        self.dihedral_unique = d_unique
        n_d = max(1, len(d_unique))
        # 3 Fourier terms (n=1, 2, 3) per type; V_n >= 0 via softplus, delta_n free
        self.dihedral_log_V = nn.Parameter(torch.full((n_d, 3), -2.0))
        self.dihedral_delta = nn.Parameter(torch.zeros((n_d, 3)))
        self._has_dihedrals = bool(dihedrals)

        # Global energy offset. All bonded terms are non-negative, but the
        # true total energy is *below* sum(E0[Z_i]) because bonding stabilises
        # the molecule. Classical FFs absorb that gap into the parameter set;
        # we just learn one scalar that anchors the absolute zero of energy.
        # Initialised from training-set statistics to avoid spending a long
        # warmup phase moving a single scalar against a steep gradient.
        self.energy_offset = nn.Parameter(torch.tensor(float(offset_init)))

        # E0 baselines
        elements = sorted(atomic_energies.keys())
        E0 = torch.tensor([atomic_energies[z] for z in elements], dtype=torch.get_default_dtype())
        self.elements = elements
        self.register_buffer("E0", E0)
        zmax = max(elements)
        z_to_idx = torch.full((zmax + 1,), -1, dtype=torch.long)
        for i, z in enumerate(elements):
            z_to_idx[z] = i
        self.register_buffer("_z_to_idx", z_to_idx)

    # ---------- per-term energy contributions ----------
    #
    # All three accept positions of shape (B, N, 3) and return a (B,) tensor.
    # Bond/angle/dihedral atom indices are fixed at construction time, so the
    # same gather works for every frame in the batch — we just `positions[:, i]`
    # instead of `positions[i]`.

    def _bond_energy(self, positions: torch.Tensor) -> torch.Tensor:
        i, j = self.bond_idx[:, 0], self.bond_idx[:, 1]
        r = torch.linalg.norm(positions[:, i] - positions[:, j] + 1e-12, dim=-1)   # (B, n_bonds)
        k = torch.nn.functional.softplus(self.bond_log_k)[self.bond_type]
        r0 = self.bond_r0[self.bond_type]
        return (k * (r - r0).pow(2)).sum(dim=-1)

    def _angle_energy(self, positions: torch.Tensor) -> torch.Tensor:
        i, j, k = self.angle_idx[:, 0], self.angle_idx[:, 1], self.angle_idx[:, 2]
        v1 = positions[:, i] - positions[:, j]          # (B, n_angles, 3)
        v2 = positions[:, k] - positions[:, j]
        cos_t = (v1 * v2).sum(-1) / (v1.norm(dim=-1) * v2.norm(dim=-1) + 1e-12)
        cos_t = cos_t.clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_t)                       # (B, n_angles)
        ka = torch.nn.functional.softplus(self.angle_log_k)[self.angle_type]
        t0 = self.angle_theta0[self.angle_type]
        return (ka * (theta - t0).pow(2)).sum(dim=-1)

    def _dihedral_energy(self, positions: torch.Tensor) -> torch.Tensor:
        B = positions.shape[0]
        if not self._has_dihedrals:
            return torch.zeros((B,), dtype=positions.dtype, device=positions.device)
        i, j, k, l = (self.dihedral_idx[:, c] for c in range(4))
        b1 = positions[:, j] - positions[:, i]          # (B, n_dih, 3)
        b2 = positions[:, k] - positions[:, j]
        b3 = positions[:, l] - positions[:, k]
        n1 = torch.cross(b1, b2, dim=-1)
        n2 = torch.cross(b2, b3, dim=-1)
        m1 = torch.cross(n1, b2 / (b2.norm(dim=-1, keepdim=True) + 1e-12), dim=-1)
        x = (n1 * n2).sum(-1)
        y = (m1 * n2).sum(-1)
        phi = torch.atan2(y, x)                         # (B, n_dih)

        V = torch.nn.functional.softplus(self.dihedral_log_V)[self.dihedral_type]   # (n_dih, 3)
        delta = self.dihedral_delta[self.dihedral_type]
        n_vals = torch.tensor([1.0, 2.0, 3.0], dtype=positions.dtype, device=positions.device)
        per_term = V * (1.0 + torch.cos(n_vals * phi.unsqueeze(-1) - delta))         # (B, n_dih, 3)
        return per_term.sum(dim=(-2, -1))

    def _energy(self, positions: torch.Tensor, z_indices: torch.Tensor) -> torch.Tensor:
        """Total energy for a batch of frames. positions (B, N, 3) -> (B,)."""
        e_bond = self._bond_energy(positions)
        e_angle = self._angle_energy(positions)
        e_dihedral = self._dihedral_energy(positions)
        e0 = self.E0[z_indices].sum(dim=-1)
        return e_bond + e_angle + e_dihedral + e0 + self.energy_offset

    def predict(self, frames: Sequence[Frame]):
        """Same homogeneous-N fast path as pair_morse: stack the batch, do one
        forward + autograd. Falls back to a per-frame loop for mixed-N data
        (which workshop1 doesn't ship — bonded_ff topology is inferred from
        one molecule, so heterogeneity is already a stretch)."""
        n_atoms = frames[0].n_atoms
        if all(fr.n_atoms == n_atoms for fr in frames):
            positions = torch.stack([fr.positions for fr in frames]).detach().requires_grad_(True)
            z_indices = torch.stack([self._z_to_idx[fr.atomic_numbers] for fr in frames])
            energies = self._energy(positions, z_indices)
            grads = torch.autograd.grad(energies.sum(), positions,
                                        create_graph=self.training)[0]
            forces = [-grads[i] for i in range(len(frames))]
            return {"energies": energies, "forces": forces}

        energies_list: List[torch.Tensor] = []
        forces: List[torch.Tensor] = []
        for fr in frames:
            z_idx = self._z_to_idx[fr.atomic_numbers]
            positions = fr.positions.detach().requires_grad_(True)
            E = self._energy(positions.unsqueeze(0), z_idx.unsqueeze(0))[0]
            F_pred = -torch.autograd.grad(E, positions, create_graph=self.training)[0]
            energies_list.append(E)
            forces.append(F_pred)
        return {"energies": torch.stack(energies_list), "forces": forces}


def build_bonded_ff(
    *,
    elements: Sequence[int],
    atomic_energies: Dict[int, float],
    r_max: float,                       # ignored — bonded FF has no r_max
    frames_for_init: Sequence[Frame] | None,
    bond_scale: float = 1.2,
    **kwargs,
) -> BondedFF:
    if not frames_for_init:
        raise ValueError("bonded_ff needs frames_for_init to infer topology from the first frame")
    # Initialise the energy offset to (mean E_total - sum E0) over the training
    # set, so the scalar starts near its target and Adam can fine-tune from
    # there rather than crawling up from zero.
    e0_per_frame = sum(atomic_energies[int(z)] for z in frames_for_init[0].atomic_numbers.tolist())
    # Single GPU->CPU transfer instead of N syncs from a per-frame float() loop;
    # this loop was the dominant setup cost on HIP for any non-tiny dataset.
    e_mean = float(torch.stack([fr.energy for fr in frames_for_init]).mean())
    offset_init = e_mean - e0_per_frame
    return BondedFF(
        ref_frame=frames_for_init[0],
        atomic_energies=atomic_energies,
        bond_scale=bond_scale,
        offset_init=offset_init,
    )
