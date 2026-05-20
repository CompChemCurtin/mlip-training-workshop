"""Adapter that exposes the existing workshop1 MACE builder via the
common `predict(frames)` interface.

The MACE machinery (`workshop1.model.build_model`, `workshop1.data`)
consumes AtomicData graphs in batches, not raw Frame objects. This
adapter:

    - converts `frames_for_init` into a one-shot DataLoader so the
      ScaleShiftMACE constructor can compute `avg_num_neighbors` and the
      per-atom-interaction-energy mean/std
    - on every `predict()` call, builds AtomicData graphs from the input
      frames, batches them, runs them through MACE, and unpacks per-frame
      energies / forces

There is some per-call graph construction cost (one AtomicData.from_config
per frame). For ethanol (9 atoms, r_max ~5 A) that's well under a
millisecond per frame; for larger systems you would want to cache the
graphs after the first construction.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn

from mace.data import AtomicData
from mace.data.utils import Configuration
from mace.tools import AtomicNumberTable, torch_geometric

from workshop1.frames import Frame
from workshop1.model import ModelConfig, build_model as build_mace_core


def _frame_to_atomic_data(fr: Frame, z_table: AtomicNumberTable, r_max: float,
                          heads: List[str]) -> AtomicData:
    """Wrap one Frame as a MACE Configuration -> AtomicData graph."""
    config = Configuration(
        atomic_numbers=fr.atomic_numbers.cpu().numpy(),
        positions=fr.positions.detach().cpu().numpy(),
        properties={
            "energy": float(fr.energy.detach().cpu()),
            "forces": fr.forces.detach().cpu().numpy(),
        },
        property_weights={"energy": 1.0, "forces": 1.0},
        cell=None,
        pbc=(False, False, False),
        head=heads[0],
    )
    return AtomicData.from_config(config, z_table=z_table, cutoff=r_max, heads=heads)


class MaceAdapter(nn.Module):
    """ScaleShiftMACE behind a predict(frames) interface."""

    def __init__(self, core: nn.Module, z_table: AtomicNumberTable,
                 r_max: float, heads: List[str]):
        super().__init__()
        self.core = core
        self.z_table = z_table
        self.r_max = float(r_max)
        self.heads = heads

    def predict(self, frames: Sequence[Frame]) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
        data_list = [_frame_to_atomic_data(fr, self.z_table, self.r_max, self.heads) for fr in frames]
        batch = torch_geometric.Batch.from_data_list(data_list)
        # AtomicData.from_config goes through numpy, so the batch lands on
        # CPU regardless of where the input frames lived. Move it to the
        # core's device so the forward pass works on GPU/MPS too.
        device = next(self.core.parameters()).device
        batch = batch.to(device)
        out = self.core(
            batch.to_dict(),
            training=self.training,
            compute_force=True,
            compute_stress=False,
        )
        # split per-frame forces using batch.ptr
        ptr = batch.ptr.cpu().tolist()
        forces_per_frame = [out["forces"][ptr[i]:ptr[i + 1]] for i in range(len(frames))]
        return {"energies": out["energy"], "forces": forces_per_frame}


def build_mace(
    *,
    elements: Sequence[int],
    atomic_energies: Dict[int, float],
    r_max: float,
    frames_for_init: Sequence[Frame] | None,
    num_interactions: int = 2,
    hidden_irreps: str = "16x0e + 16x1o",
    correlation: int = 3,
    batch_size_for_init: int = 8,
    **kwargs,
) -> MaceAdapter:
    """Construct ScaleShiftMACE behind the predict(frames) interface.

    The default `hidden_irreps` here is intentionally tiny (16 scalar +
    16 vector channels) versus a production MACE's 128x0e+128x1o. We train
    several MACE runs back to back in the workshop, and force training does
    an expensive double-backward whose cost scales with model size, so we
    keep the model small enough to stay snappy. A 9-atom molecule needs
    nothing like production capacity. Pass `hidden_irreps` through
    `**kwargs` to override.
    """
    if not frames_for_init:
        raise ValueError("mace adapter needs frames_for_init to compute scale/shift stats")

    z_table = AtomicNumberTable(sorted(elements))
    heads = ["Default"]

    # Build a small DataLoader from the init frames so the existing builder
    # can compute avg_num_neighbors and the per-atom inter-energy mean/std.
    init_data = [_frame_to_atomic_data(fr, z_table, r_max, heads) for fr in frames_for_init]
    init_loader = torch_geometric.dataloader.DataLoader(
        init_data, batch_size=batch_size_for_init, shuffle=False, drop_last=False
    )

    cfg = ModelConfig(
        r_max=r_max,
        num_interactions=num_interactions,
        hidden_irreps=hidden_irreps,
        correlation=correlation,
    )
    core = build_mace_core(cfg, z_table, atomic_energies, init_loader, heads=heads)
    return MaceAdapter(core=core, z_table=z_table, r_max=r_max, heads=heads)
