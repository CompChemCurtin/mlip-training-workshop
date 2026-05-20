"""Lightweight per-frame data structure shared across the simple models.

The MACE training path (`workshop1/data.py`) builds full `AtomicData`
graphs because the GNN needs them. The simpler models in
`workshop1/models/` (pair-Morse, bonded FF, PET toy) don't need a graph
representation — they read positions and atomic numbers directly. This
module is the common ground.

A `Frame` is one configuration:
    positions       (N, 3) float
    atomic_numbers  (N,) long
    energy          scalar
    forces          (N, 3) float
    n_atoms         int

`load_ethanol` reads an extxyz file (with the workshop's REF_energy /
REF_forces convention), pulls out IsolatedAtom entries as per-element
E0 baselines, and returns (frames, atomic_energies).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from ase.io import read


@dataclass
class Frame:
    positions: torch.Tensor          # (N, 3) float
    atomic_numbers: torch.Tensor     # (N,) long
    energy: torch.Tensor             # scalar
    forces: torch.Tensor             # (N, 3)
    n_atoms: int


def load_ethanol(xyz_path: Path, dtype: torch.dtype = torch.float64) -> Tuple[List[Frame], Dict[int, float]]:
    """Load an rMD17-style extxyz file.

    IsolatedAtom frames are recognised by `info["config_type"]` and
    contribute one per-element E0 each. All other frames are returned as
    training/validation data.
    """
    atoms_list = read(str(xyz_path), index=":")
    atomic_energies: Dict[int, float] = {}
    frames: List[Frame] = []
    for a in atoms_list:
        if a.info.get("config_type") == "IsolatedAtom":
            atomic_energies[int(a.numbers[0])] = float(a.info["REF_energy"])
            continue
        frames.append(Frame(
            positions=torch.as_tensor(a.get_positions(), dtype=dtype),
            atomic_numbers=torch.as_tensor(a.numbers, dtype=torch.long),
            energy=torch.as_tensor(a.info["REF_energy"], dtype=dtype),
            forces=torch.as_tensor(a.arrays["REF_forces"], dtype=dtype),
            n_atoms=len(a),
        ))
    return frames, atomic_energies


def split(frames: Sequence[Frame], valid_fraction: float, seed: int) -> Tuple[List[Frame], List[Frame]]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(frames))
    n_val = max(1, int(round(len(frames) * valid_fraction)))
    return [frames[i] for i in idx[n_val:]], [frames[i] for i in idx[:n_val]]


def rotate(frame: Frame, R: torch.Tensor) -> Frame:
    """Return a Frame with positions and forces rotated by R (3x3).

    Energies and atomic numbers are invariant. Used by PET's data
    augmentation hook (`workshop1.models.pet.PETAdapter.augment`).
    """
    R = R.to(dtype=frame.positions.dtype)
    return Frame(
        positions=frame.positions @ R.T,
        atomic_numbers=frame.atomic_numbers,
        energy=frame.energy,
        forces=frame.forces @ R.T,
        n_atoms=frame.n_atoms,
    )
