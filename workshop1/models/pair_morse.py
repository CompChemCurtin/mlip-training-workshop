"""Pair-only learnable Morse potential.

The simplest legitimate interatomic model: every atom pair (i, j) within
`r_max` contributes a Morse term parameterised by the unordered element
pair (Z_i, Z_j). Energy is summed over all pairs; forces fall out of
autograd of E w.r.t. positions.

For ethanol (elements H, C, O) there are 3*(3+1)/2 = 6 element pairs
(HH, HC, HO, CC, CO, OO). Each Morse has 3 parameters (D, a, r0), so
18 learnable parameters total. This model has no angular awareness
whatsoever — it is the strawman that motivates everything that comes
after.

Form:
    E_pair(r) = D [1 - exp(-a (r - r0))]^2 - D       (D, a > 0)
    E_inter   = (1/2) sum_{i,j: i!=j, r_ij < r_max} E_pair(r_ij)
    E_total   = E_inter + sum_i E0[Z_i]

`D` and `a` are stored as their softplus pre-images so they stay
positive throughout training; `r0` is unconstrained. The `-D` shift
makes the Morse dissociate to 0 at large r so the atomic-energy
baseline E0 carries the absolute level.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from workshop1.frames import Frame


def _pair_index_table(n_elements: int) -> torch.Tensor:
    """Return an (n, n) integer table mapping (zi_idx, zj_idx) -> pair_idx.

    Pair indexing flattens the upper triangle (with diagonal):
        idx(i, j) = i*n - i*(i-1)/2 + (j - i)    for i <= j
    Order: (0,0), (0,1), ..., (0,n-1), (1,1), (1,2), ..., (n-1,n-1).
    """
    table = torch.zeros((n_elements, n_elements), dtype=torch.long)
    for i in range(n_elements):
        for j in range(n_elements):
            lo, hi = (i, j) if i <= j else (j, i)
            table[i, j] = lo * n_elements - lo * (lo - 1) // 2 + (hi - lo)
    return table


def _z_to_idx_lookup(elements: Sequence[int]) -> torch.Tensor:
    """Return a (Zmax+1,) long tensor mapping atomic number -> position in `elements`."""
    zmax = max(elements)
    table = torch.full((zmax + 1,), -1, dtype=torch.long)
    for i, z in enumerate(elements):
        table[z] = i
    return table


class PairMorse(nn.Module):
    """Sum of element-pair Morse potentials with autograd forces."""

    def __init__(
        self,
        elements: Sequence[int],
        atomic_energies: Dict[int, float],
        r_max: float = 5.0,
        D_init: float = 0.5,
        a_init: float = 2.0,
        r0_init: float = 1.5,
    ):
        super().__init__()
        self.elements = list(elements)
        n = len(self.elements)
        n_pairs = n * (n + 1) // 2
        self.r_max = float(r_max)

        # Keep D, a strictly positive via softplus reparameterisation. We
        # initialise log_D / log_a so softplus(log_D) ~ D_init.
        inv_softplus = lambda x: float(torch.log(torch.expm1(torch.tensor(x))))
        self.log_D = nn.Parameter(torch.full((n_pairs,), inv_softplus(D_init)))
        self.log_a = nn.Parameter(torch.full((n_pairs,), inv_softplus(a_init)))
        self.r0 = nn.Parameter(torch.full((n_pairs,), float(r0_init)))

        E0 = torch.tensor([atomic_energies[z] for z in self.elements], dtype=torch.get_default_dtype())
        self.register_buffer("E0", E0)
        self.register_buffer("_pair_table", _pair_index_table(n))
        self.register_buffer("_z_to_idx", _z_to_idx_lookup(self.elements))

    @property
    def D(self) -> torch.Tensor:
        return F.softplus(self.log_D)

    @property
    def a(self) -> torch.Tensor:
        return F.softplus(self.log_a)

    def _energy(self, positions: torch.Tensor, z_indices: torch.Tensor) -> torch.Tensor:
        """Total energy for a batch of frames.

        Shapes: positions (B, N, 3), z_indices (B, N) -> energy (B,).
        `positions` must have requires_grad=True if you want forces.
        """
        B, N, _ = positions.shape
        rij = positions.unsqueeze(2) - positions.unsqueeze(1)         # (B, N, N, 3)
        d = torch.linalg.norm(rij + 1e-12, dim=-1)                    # (B, N, N)
        mask = (d > 1e-6) & (d < self.r_max)

        zi = z_indices.unsqueeze(2).expand(B, N, N)
        zj = z_indices.unsqueeze(1).expand(B, N, N)
        pair_idx = self._pair_table[zi, zj]                           # (B, N, N)

        D = self.D[pair_idx]
        a = self.a[pair_idx]
        r0 = self.r0[pair_idx]
        morse = D * (1.0 - torch.exp(-a * (d - r0))) ** 2 - D
        pair_energy = 0.5 * torch.where(mask, morse, torch.zeros_like(morse)).sum(dim=(-2, -1))
        e0_sum = self.E0[z_indices].sum(dim=-1)                       # (B,)
        return pair_energy + e0_sum

    def predict(self, frames: Sequence[Frame]) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        """Predict energy and forces for a batch of frames.

        When every frame has the same atom count (the workshop's ethanol case
        and most homogeneous-molecule datasets), we stack positions into one
        `(B, N, 3)` tensor and do *one* forward + autograd call across the
        whole batch. The grad of `sum(energies)` w.r.t. the stacked positions
        is block-diagonal across the batch dimension (E_b doesn't depend on
        r_a for a != b), so a single call gives us per-frame forces too.

        For heterogeneous N (mixed molecules), we fall back to a per-frame
        loop — the readable textbook implementation.

        Returns:
            {"energies": (B,) tensor of total energies,
             "forces":   list of (N_i, 3) force tensors, one per frame}
        """
        n_atoms = frames[0].n_atoms
        if all(fr.n_atoms == n_atoms for fr in frames):
            positions = torch.stack([fr.positions for fr in frames])         # (B, N, 3)
            z_indices = torch.stack([self._z_to_idx[fr.atomic_numbers]
                                     for fr in frames])                      # (B, N)
            positions = positions.detach().requires_grad_(True)
            energies = self._energy(positions, z_indices)                    # (B,)
            grads = torch.autograd.grad(energies.sum(), positions,
                                        create_graph=self.training)[0]       # (B, N, 3)
            forces = [-grads[i] for i in range(len(frames))]
            return {"energies": energies, "forces": forces}

        # Heterogeneous-N fallback: per-frame loop.
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


def build_pair_morse(
    *,
    elements: Sequence[int],
    atomic_energies: Dict[int, float],
    r_max: float,
    frames_for_init=None,  # ignored: pair-Morse needs no data-derived stats
    **kwargs,
) -> PairMorse:
    return PairMorse(elements=elements, atomic_energies=atomic_energies, r_max=r_max, **kwargs)
