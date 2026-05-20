"""Sanity-check a trained MACE / PET model against held-out reference data.

Loads the model via `workshop2.evaluate.calculator.load_calculator`, runs
it over every non-IsolatedAtom frame in `--xyz`, and reports:

    - per-atom energy MAE/RMSE/max
    - per-component force MAE/RMSE/max

`--xyz` is normally the test split produced by `data/split_xyz.py`. The
reference keys default to the workshop1 convention (`REF_energy` /
`REF_forces`); pass `--energy-key energy --forces-key forces` for the
MAD-style bare keys.

    python -m workshop2.evaluate.ase_sanity \\
        --model runs/smoke_ethanol_mace/checkpoints/smoke_ethanol_mace_run-1234.model \\
        --xyz data/ethanol_subset.xyz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import ase.io
import numpy as np

from workshop2.evaluate.calculator import load_calculator, select_device


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True,
                   help="Path to a trained .model (MACE) or .pt (PET/metatomic).")
    p.add_argument("--xyz", type=Path, required=True,
                   help="Held-out extxyz file with REF energies / forces.")
    p.add_argument("--energy-key", type=str, default="REF_energy",
                   help="Frame.info key for the reference energy (default REF_energy).")
    p.add_argument("--forces-key", type=str, default="REF_forces",
                   help="Frame.arrays key for the reference forces (default REF_forces).")
    p.add_argument("--device", type=str, default="auto",
                   help="auto | cuda | cuda:0 | cpu  (default: auto).")
    p.add_argument("--limit", type=int, default=None,
                   help="Only evaluate the first N usable frames (smoke speed).")
    args = p.parse_args()
    device = select_device(args.device)

    print(f"loading {args.model} on {device}")
    calc = load_calculator(args.model, device=device)

    frames: List = []
    for atoms in ase.io.iread(str(args.xyz)):
        if atoms.info.get("config_type") == "IsolatedAtom":
            continue
        frames.append(atoms)
        if args.limit is not None and len(frames) >= args.limit:
            break
    print(f"evaluating {len(frames)} frames from {args.xyz}")

    e_errs_per_atom: List[float] = []
    f_errs: List[np.ndarray] = []
    t0 = time.time()
    for atoms in frames:
        ref_E = float(atoms.info[args.energy_key])
        ref_F = np.asarray(atoms.arrays[args.forces_key], dtype=np.float64)
        atoms.calc = calc
        pred_E = float(atoms.get_potential_energy())
        pred_F = np.asarray(atoms.get_forces(), dtype=np.float64)
        e_errs_per_atom.append((pred_E - ref_E) / len(atoms))
        f_errs.append((pred_F - ref_F).ravel())
    elapsed = time.time() - t0

    e_err = np.asarray(e_errs_per_atom)
    f_err = np.concatenate(f_errs)
    print(f"  done in {elapsed:.1f}s "
          f"({1e3 * elapsed / max(1, len(frames)):.1f} ms/frame)")

    print()
    print(f"per-atom energy error (eV/atom):")
    print(f"  RMSE = {np.sqrt(np.mean(e_err ** 2)) * 1e3:8.2f} meV/atom")
    print(f"  MAE  = {np.mean(np.abs(e_err)) * 1e3:8.2f} meV/atom")
    print(f"  max  = {np.max(np.abs(e_err)) * 1e3:8.2f} meV/atom")
    print()
    print(f"per-component force error (eV/A):")
    print(f"  RMSE = {np.sqrt(np.mean(f_err ** 2)) * 1e3:8.2f} meV/A")
    print(f"  MAE  = {np.mean(np.abs(f_err)) * 1e3:8.2f} meV/A")
    print(f"  max  = {np.max(np.abs(f_err)) * 1e3:8.2f} meV/A")


if __name__ == "__main__":
    main()
