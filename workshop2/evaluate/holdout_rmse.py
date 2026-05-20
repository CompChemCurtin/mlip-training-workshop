"""Side-by-side held-out RMSE table across multiple trained checkpoints.

Generalisation of `ase_sanity.py`: same evaluation, run on N models at
once, printed as one comparison table. Handy for picking between MACE
configurations, between MACE and PET, or between SWA vs non-SWA tails of
the same training.

    python -m workshop2.evaluate.holdout_rmse \\
        --model runs/.../mace_run-1234.model     --label mace \\
        --model runs/.../mace_run-1234_swa.model --label mace+swa \\
        --model runs/smoke_ethanol_pet/model.pt  --label pet \\
        --xyz data/ethanol_subset.xyz

Repeat `--model <path> --label <name>` for each checkpoint. `--label`
defaults to the checkpoint stem when omitted, but you'll usually want to
supply one explicitly for the table.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import ase.io
import numpy as np

from workshop2.evaluate.calculator import load_calculator, select_device


def evaluate_one(model_path: Path, frames, *, energy_key: str,
                 forces_key: str, device: str) -> Dict[str, float]:
    calc = load_calculator(model_path, device=device)
    e_errs: List[float] = []
    f_errs: List[np.ndarray] = []
    t0 = time.time()
    for atoms in frames:
        ref_E = float(atoms.info[energy_key])
        ref_F = np.asarray(atoms.arrays[forces_key], dtype=np.float64)
        atoms.calc = calc
        pred_E = float(atoms.get_potential_energy())
        pred_F = np.asarray(atoms.get_forces(), dtype=np.float64)
        e_errs.append((pred_E - ref_E) / len(atoms))
        f_errs.append((pred_F - ref_F).ravel())
    elapsed = time.time() - t0
    e_arr = np.asarray(e_errs)
    f_arr = np.concatenate(f_errs)
    return {
        "rmse_E": float(np.sqrt(np.mean(e_arr ** 2)) * 1e3),
        "mae_E":  float(np.mean(np.abs(e_arr)) * 1e3),
        "rmse_F": float(np.sqrt(np.mean(f_arr ** 2)) * 1e3),
        "mae_F":  float(np.mean(np.abs(f_arr)) * 1e3),
        "time_s": elapsed,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", action="append", required=True, type=Path, dest="models",
                   help="Trained checkpoint. Repeat for each one.")
    p.add_argument("--label", action="append", default=None, dest="labels")
    p.add_argument("--xyz", type=Path, required=True)
    p.add_argument("--energy-key", type=str, default="REF_energy")
    p.add_argument("--forces-key", type=str, default="REF_forces")
    p.add_argument("--limit", type=int, default=None,
                   help="Only evaluate the first N usable frames.")
    p.add_argument("--device", type=str, default="auto",
                   help="auto | cuda | cuda:0 | cpu  (default: auto).")
    args = p.parse_args()
    device = select_device(args.device)

    labels = args.labels or [m.stem for m in args.models]
    if len(labels) != len(args.models):
        raise SystemExit(
            f"got {len(args.models)} --model but {len(labels)} --label; "
            "supply the same number of each or omit --label entirely."
        )

    frames = []
    for atoms in ase.io.iread(str(args.xyz)):
        if atoms.info.get("config_type") == "IsolatedAtom":
            continue
        frames.append(atoms)
        if args.limit is not None and len(frames) >= args.limit:
            break
    print(f"evaluating {len(frames)} frames from {args.xyz}\n")

    rows: List[Dict[str, object]] = []
    for model_path, label in zip(args.models, labels):
        print(f">>> {label}  ({model_path})")
        m = evaluate_one(
            model_path, frames,
            energy_key=args.energy_key, forces_key=args.forces_key,
            device=device,
        )
        print(f"  done in {m['time_s']:.1f}s  "
              f"(RMSE_E={m['rmse_E']:.2f} meV/atom, RMSE_F={m['rmse_F']:.1f} meV/A)")
        rows.append({"label": label, **m})

    _print_table(rows)


def _print_table(rows: List[Dict[str, object]]) -> None:
    header = (
        f"{'label':<20} {'time (s)':>10} "
        f"{'RMSE_E meV/atom':>16} {'MAE_E meV/atom':>16} "
        f"{'RMSE_F meV/A':>14} {'MAE_F meV/A':>14}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['label']:<20} "
            f"{r['time_s']:>10.1f} "
            f"{r['rmse_E']:>16.2f} "
            f"{r['mae_E']:>16.2f} "
            f"{r['rmse_F']:>14.1f} "
            f"{r['mae_F']:>14.1f}"
        )


if __name__ == "__main__":
    main()
