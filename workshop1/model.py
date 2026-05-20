"""Construct a MACE model from its components.

A MACE model is, conceptually:

    1. Radial embedding R(r)        -- Bessel basis * polynomial cutoff
    2. Spherical harmonics Y_lm(r)  -- angular features up to max_ell
    3. N message-passing layers     -- equivariant tensor products of
                                       node features with edge features,
                                       followed by a many-body symmetric
                                       contraction of order `correlation`
    4. Linear/MLP readouts          -- per-layer node-energy heads
    5. Scale-shift and atomic E0s   -- final per-atom energy =
                                       scale * sum_layers(node_energy_l)
                                       + shift + E0[Z_i]

The total energy is the sum of node energies; forces and stress are
obtained by autograd of E w.r.t. positions and the strain tensor,
which the MACE module handles internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
from e3nn import o3

from mace.modules import (
    MACE,
    RealAgnosticResidualInteractionBlock,
    ScaleShiftMACE,
)
from mace.modules.utils import (
    compute_avg_num_neighbors,
    compute_mean_std_atomic_inter_energy,
)
from mace.tools import AtomicNumberTable


@dataclass
class ModelConfig:
    r_max: float = 5.0                       # cutoff (must match data)
    num_bessel: int = 8                      # radial basis size
    num_polynomial_cutoff: int = 5           # cutoff envelope order
    max_ell: int = 3                         # angular order of Y_lm
    num_interactions: int = 2                # number of message-passing layers
    correlation: int = 3                     # body-order at each layer (3 -> 4-body features)
    hidden_irreps: str = "128x0e + 128x1o"   # invariant + vector channels per node
    mlp_irreps: str = "16x0e"                # readout MLP hidden width
    radial_mlp: tuple = (64, 64, 64)         # MLP turning radial basis into per-edge weights


def _atomic_energies_array(e0s: Dict[int, float], z_table: AtomicNumberTable) -> np.ndarray:
    """Order the per-element baselines to match z_table indexing."""
    return np.array([e0s[z] for z in z_table.zs], dtype=np.float64)


def build_model(
    cfg: ModelConfig,
    z_table: AtomicNumberTable,
    atomic_energies: Dict[int, float],
    train_loader,
    heads: List[str] | None = None,
) -> torch.nn.Module:
    """Build a freshly initialised ScaleShiftMACE.

    Two dataset-derived statistics enter the model:

    - `avg_num_neighbors` normalises the message sum so that activations
      do not blow up with denser graphs.
    - `mean`, `std` of per-atom interaction energies (after subtracting
      E0 baselines) become the readout shift / scale, so the network's
      raw output starts in O(1).

    Both are computed on the training loader once and frozen.
    """
    e0_array = _atomic_energies_array(atomic_energies, z_table)

    avg_num_neighbors = compute_avg_num_neighbors(train_loader)
    mean, std = compute_mean_std_atomic_inter_energy(train_loader, e0_array)

    heads = heads or ["Default"]

    model = ScaleShiftMACE(
        # --- 1. radial / angular bases ---
        r_max=cfg.r_max,
        num_bessel=cfg.num_bessel,
        num_polynomial_cutoff=cfg.num_polynomial_cutoff,
        max_ell=cfg.max_ell,
        radial_MLP=list(cfg.radial_mlp),
        # --- 2. message-passing stack ---
        interaction_cls_first=RealAgnosticResidualInteractionBlock,
        interaction_cls=RealAgnosticResidualInteractionBlock,
        num_interactions=cfg.num_interactions,
        correlation=cfg.correlation,
        hidden_irreps=o3.Irreps(cfg.hidden_irreps),
        MLP_irreps=o3.Irreps(cfg.mlp_irreps),
        gate=torch.nn.functional.silu,
        # --- 3. element bookkeeping & baselines ---
        num_elements=len(z_table),
        atomic_numbers=list(z_table.zs),
        atomic_energies=e0_array,
        avg_num_neighbors=avg_num_neighbors,
        # --- 4. final scale/shift on the learned interaction energy ---
        atomic_inter_scale=float(std),
        atomic_inter_shift=float(mean),
        heads=heads,
    )
    return model


def n_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
