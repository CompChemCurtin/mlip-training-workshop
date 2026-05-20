"""Adapter for the upstream PET implementation in `metatrain`.

PET (Point-Edge Transformer, Pozdnyakov & Ceriotti 2023) is a transformer
over atom-centred edge tokens that *learns* approximate rotation
invariance from data augmentation rather than building it in like MACE.
This module wraps `metatrain.pet.PET` so it conforms to the same
`predict(frames)` contract as the other workshop models — fixed-topology
classical FF, pair-Morse, and MACE — and can be slotted into the
metrics walkthrough side-by-side with them.

Mechanics:

- Per `predict()` call we convert each Frame into a `metatomic.torch.System`
  with an attached vesin neighbour list (PET requires one to be present).
- The metatrain PET model is set up to emit *interaction* energies; we
  add `sum_i E0[Z_i]` ourselves to recover totals comparable to the
  dataset's REF_energy.
- Forces are obtained by autograd on the input positions, matching how
  every other model in `workshop1/models/` produces them.

This is intentionally a *thin* wrapper — it just glues the API. The
hyperparameters here are slimmer than the metatrain defaults so the
metrics walkthrough fits in a workshop time budget; for production
runs, prefer metatrain's CLI which handles checkpointing, EMA, schedule,
DDP, etc.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import vesin
from omegaconf import OmegaConf

from metatomic.torch import ModelOutput, NeighborListOptions, System
from metatensor.torch import Labels, TensorBlock

from metatrain.pet import PET
from metatrain.pet.documentation import ModelHypers
from metatrain.utils.augmentation import get_random_inversion, get_random_rotation
from metatrain.utils.data.dataset import DatasetInfo
from metatrain.utils.data.target_info import get_energy_target_info
from metatrain.utils.hypers import init_with_defaults

from workshop1.frames import Frame, rotate


_NL_SAMPLE_NAMES = [
    "first_atom", "second_atom",
    "cell_shift_a", "cell_shift_b", "cell_shift_c",
]


def _neighbour_metadata(positions: torch.Tensor, atomic_numbers: torch.Tensor,
                        r_max: float) -> dict:
    """Compute the per-frame-*invariant* neighbour-list metadata once.

    Everything here is constant across frames that share a molecule: the
    (i, j) edge indices, the metatensor Labels, the NeighborListOptions,
    and --- crucially for GPU throughput --- the `types`, `cell` and `pbc`
    tensors. Building these per frame meant ~4 tiny GPU tensor allocations
    times batch size every step; on an MI250X that per-frame launch
    overhead dominated. We build them once and cache them on the adapter.
    Only the edge vectors change per frame, and (see predict) even those
    are computed for the whole batch in a single op.
    """
    device = positions.device
    pts = positions.detach().cpu().numpy().astype(np.float64)
    nl = vesin.NeighborList(cutoff=r_max, full_list=True)
    i, j, S = nl.compute(
        points=pts, box=np.zeros((3, 3)), periodic=False, quantities="ijS"
    )
    samples = Labels(
        names=_NL_SAMPLE_NAMES,
        values=torch.stack([
            torch.as_tensor(i,       dtype=torch.int32, device=device),
            torch.as_tensor(j,       dtype=torch.int32, device=device),
            torch.as_tensor(S[:, 0], dtype=torch.int32, device=device),
            torch.as_tensor(S[:, 1], dtype=torch.int32, device=device),
            torch.as_tensor(S[:, 2], dtype=torch.int32, device=device),
        ]).T,
    )
    return {
        "i_idx": torch.as_tensor(i, dtype=torch.long, device=device),
        "j_idx": torch.as_tensor(j, dtype=torch.long, device=device),
        "samples": samples,
        "components": [Labels(
            names=["xyz"],
            values=torch.tensor([[0], [1], [2]], dtype=torch.int32, device=device),
        )],
        "properties": Labels(
            names=["distance"],
            values=torch.tensor([[0]], dtype=torch.int32, device=device),
        ),
        "options": NeighborListOptions(cutoff=r_max, full_list=True, strict=True),
        "types": atomic_numbers.to(torch.int32),
        "cell": torch.zeros((3, 3), dtype=positions.dtype, device=device),
        "pbc": torch.tensor([False, False, False], device=device),
    }


def _make_system(positions: torch.Tensor, distances: torch.Tensor,
                 meta: dict) -> System:
    """Build a metatomic System from positions, precomputed edge vectors,
    and cached neighbour metadata.

    `positions` and `distances` share storage with the leaf tensor the
    caller is autograd'ing against. Everything else (types, cell, pbc, the
    Labels, the options) comes from `meta` --- no per-frame tensor
    allocation, so this is just metatomic object construction.
    """
    system = System(types=meta["types"], positions=positions,
                    cell=meta["cell"], pbc=meta["pbc"])
    block = TensorBlock(
        values=distances,
        samples=meta["samples"],
        components=meta["components"],
        properties=meta["properties"],
    )
    system.add_neighbor_list(meta["options"], block)
    return system


class PETAdapter(nn.Module):
    """metatrain PET behind the workshop's predict(frames) interface."""

    def __init__(self, core: PET, atomic_energies: Dict[int, float], r_max: float,
                 element_to_idx: torch.Tensor):
        super().__init__()
        self.core = core
        self.r_max = float(r_max)
        E0 = torch.tensor(
            [atomic_energies[int(z)] for z in sorted(atomic_energies)],
            dtype=torch.get_default_dtype(),
        )
        self.register_buffer("E0", E0)
        self.register_buffer("_z_to_idx", element_to_idx)
        self._outputs = {"energy": ModelOutput(quantity="energy", unit="eV", per_atom=False)}
        # Neighbour-list metadata cache, keyed by (n_atoms, device). A plain
        # attribute (not a buffer) so it stays out of state_dict and is
        # rebuilt lazily after load / device move.
        self._nl_cache: Dict[tuple, dict] = {}

    def _neighbour_meta(self, ref_positions: torch.Tensor,
                        ref_atomic_numbers: torch.Tensor) -> dict:
        key = (ref_positions.shape[0], str(ref_positions.device))
        meta = self._nl_cache.get(key)
        if meta is None:
            meta = _neighbour_metadata(ref_positions, ref_atomic_numbers, self.r_max)
            self._nl_cache[key] = meta
        return meta

    def augment(self, frames: Sequence[Frame]) -> List[Frame]:
        """Return rotation+inversion-augmented copies of `frames`.

        Per frame we draw an independent Haar-uniform SO(3) rotation
        (`get_random_rotation`) and an independent random sign in {+1, -1}
        (`get_random_inversion`), then build a single 3x3 transform
        `T = sign * R` whose determinant is +1 (rotation) or -1 (improper).
        Positions and forces (rank-1 tensors) both transform as
        `x -> T x`. Energy is invariant.

        This matches what `metatrain.utils.augmentation.RotationalAugmenter
        .apply_random_augmentations` does for the energy + forces case.
        """
        out: List[Frame] = []
        for fr in frames:
            R_np = get_random_rotation().as_matrix() * get_random_inversion()
            T = torch.as_tensor(R_np, dtype=fr.positions.dtype, device=fr.positions.device)
            out.append(rotate(fr, T))
        return out

    def predict(self, frames: Sequence[Frame]) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        """Batched forward + one autograd call.

        metatomic's PET accepts a list of Systems and batches them internally.
        We exploit that: stack positions once, build N systems whose
        position fields are *views* into the stacked leaf, hand the whole
        list to `self.core` in one call, and do a single autograd.grad on
        the stacked positions for forces.

        The per-frame system construction reuses cached neighbour-list
        metadata (see `_neighbour_meta`): for a fixed molecule the edge
        index structure is constant, so only the edge vectors are rebuilt
        per frame. This is what makes batches of a few hundred frames cheap.
        The homogeneous-N path shares one metadata object across the batch;
        the mixed-N fallback caches per atom count.
        """
        n_atoms = frames[0].n_atoms
        homogeneous = all(fr.n_atoms == n_atoms for fr in frames)

        if homogeneous:
            positions_stacked = torch.stack([fr.positions for fr in frames])
            positions_stacked = positions_stacked.detach().requires_grad_(True)
            meta = self._neighbour_meta(positions_stacked[0], frames[0].atomic_numbers)
            # All edge vectors for the whole batch in ONE op, then sliced per
            # system. Replaces B separate gather+subtract launches -- the bulk
            # of the per-frame GPU overhead on the build-systems phase.
            i_idx, j_idx = meta["i_idx"], meta["j_idx"]
            edge_vecs = (positions_stacked[:, j_idx] - positions_stacked[:, i_idx]).unsqueeze(-1)
            systems = [
                _make_system(positions_stacked[b], edge_vecs[b], meta)
                for b in range(len(frames))
            ]
            result = self.core(systems, self._outputs)
            E_inter = result["energy"].block().values.reshape(-1)         # (B,)
            z_indices = torch.stack([self._z_to_idx[fr.atomic_numbers] for fr in frames])
            E_total = E_inter + self.E0[z_indices].sum(dim=-1)            # (B,)
            grads = torch.autograd.grad(E_total.sum(), positions_stacked,
                                        create_graph=self.training)[0]
            forces = [-grads[i] for i in range(len(frames))]
            return {"energies": E_total, "forces": forces}

        # Heterogeneous-N fallback: per-frame loop, but still one model call.
        positions_list = [fr.positions.detach().clone().requires_grad_(True) for fr in frames]
        systems = []
        for p, fr in zip(positions_list, frames):
            meta = self._neighbour_meta(p, fr.atomic_numbers)
            distances = (p[meta["j_idx"]] - p[meta["i_idx"]]).unsqueeze(-1)
            systems.append(_make_system(p, distances, meta))
        result = self.core(systems, self._outputs)
        E_inter = result["energy"].block().values.reshape(-1)
        E_total_list = [
            E_inter[i] + self.E0[self._z_to_idx[fr.atomic_numbers]].sum()
            for i, fr in enumerate(frames)
        ]
        E_total = torch.stack(E_total_list)
        forces = [
            -torch.autograd.grad(E_total[i], positions_list[i],
                                 create_graph=self.training,
                                 retain_graph=(i < len(frames) - 1))[0]
            for i in range(len(frames))
        ]
        return {"energies": E_total, "forces": forces}


def build_pet(
    *,
    elements: Sequence[int],
    atomic_energies: Dict[int, float],
    r_max: float,
    frames_for_init=None,         # not strictly required
    d_pet: int = 32,
    d_node: int = 32,
    d_head: int = 32,
    d_feedforward: int = 64,
    num_attention_layers: int = 1,
    num_gnn_layers: int = 1,
    num_heads: int = 4,
    **kwargs,
) -> PETAdapter:
    """Construct a fresh PET model (slim hypers) behind the predict(frames) API.

    Defaults are deliberately tiny (d_pet=d_node=32, one GNN + one attention
    layer) versus a production PET (d_pet=128+, several layers). Force
    training does a double-backward that scales with model size, and a
    9-atom molecule needs nothing like production capacity. Override via
    `**kwargs` for larger systems.
    """
    elements_sorted = sorted(int(z) for z in elements)
    target_cfg = OmegaConf.create({
        "quantity": "energy", "unit": "eV", "per_atom": False,
        "num_subtargets": 1, "type": "scalar",
    })
    target_info = get_energy_target_info(
        "energy", target_cfg, add_position_gradients=True
    )
    dataset_info = DatasetInfo(
        length_unit="Angstrom",
        atomic_types=elements_sorted,
        targets={"energy": target_info},
    )

    hypers = init_with_defaults(ModelHypers)
    hypers["cutoff"] = float(r_max)
    hypers["d_pet"] = d_pet
    hypers["d_node"] = d_node
    hypers["d_head"] = d_head
    hypers["d_feedforward"] = d_feedforward
    hypers["num_attention_layers"] = num_attention_layers
    hypers["num_gnn_layers"] = num_gnn_layers
    hypers["num_heads"] = num_heads

    core = PET(hypers, dataset_info)

    # Lookup table for atomic-number -> position in elements_sorted.
    zmax = max(elements_sorted)
    z_to_idx = torch.full((zmax + 1,), -1, dtype=torch.long)
    for i, z in enumerate(elements_sorted):
        z_to_idx[z] = i

    return PETAdapter(
        core=core,
        atomic_energies={int(z): float(atomic_energies[z]) for z in elements_sorted},
        r_max=r_max,
        element_to_idx=z_to_idx,
    )
