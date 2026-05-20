"""Train every model in the registry on the same loss recipe, side by side.

`train.py` answers "what does the loss recipe buy me?" by fixing the model
and varying the loss. This script answers the orthogonal question: "what
does the model architecture buy me?" — fixing the loss (default `E:F = 1:100`,
MACE's canonical recipe) and stepping through the four workshop models from
the simplest possible interatomic potential to a full MACE.

    python -m workshop1.compare_models

The four architectures, in order of increasing inductive bias / parameter count:

    pair_morse  — 18 params. Element-pair Morse potentials. No angles.
    bonded_ff   — ~50 params. Bonds + angles + dihedrals over a fixed topology.
    pet         — ~50k params. Point-edge transformer. Equivariance from
                  data augmentation, not construction.
    mace        — ~100k params. Equivariant by construction; many-body via
                  symmetric tensor-product contractions.

The same loss recipe is used for every model; the per-model defaults
(epochs, lr, batch size) come from `DEFAULT_HYPERPARAMS` and are tuned so
the whole sweep fits in ~5 min of laptop CPU time. Checkpoints land under
`runs/compare/<model>.pt` so `workshop1.md` can pick any of them up.

The closing table shows trainable parameters, wall-clock training time,
and validation RMSE for energy and forces. The interesting reads:

    - Compare `pair_morse` (no angular awareness) against `bonded_ff` (has
      angles via the fixed topology): how much does adding angular terms
      help on a molecule whose dynamics are dominated by bending modes?
    - Compare `bonded_ff` against `pet`/`mace`: when does the cost of a
      learned model start to pay off over a hand-coded functional form?
    - Compare `pet` against `mace`: same parameter budget, different
      equivariance strategies. Where do they end up?
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import torch

from workshop1.frames import load_ethanol, split
from workshop1.models import DEFAULT_HYPERPARAMS, available_models, build_model
from workshop1.train import (
    LossRecipe,
    device_label,
    frames_to,
    select_device,
    train_one,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=None, choices=available_models(),
                   help="Subset of models to compare (default: all four).")
    p.add_argument("--xyz", type=Path, default=Path("data/ethanol_subset.xyz"))
    p.add_argument("--out", type=Path, default=Path("runs/compare"),
                   help="Output directory for checkpoints.")
    p.add_argument("--w-E", type=float, default=1.0,
                   help="Energy loss weight (default 1.0).")
    p.add_argument("--w-F", type=float, default=100.0,
                   help="Forces loss weight (default 100.0 = MACE's canonical 1:100).")
    p.add_argument("--epochs", type=int, default=None,
                   help="Override the per-model epoch count (default: each "
                        "model's DEFAULT_HYPERPARAMS value). Useful for a "
                        "quick demo across all four at once.")
    p.add_argument("--r-max", type=float, default=5.0)
    p.add_argument("--valid-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no-save", action="store_true",
                   help="Skip writing checkpoints.")
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    args = p.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    torch.set_default_dtype(dtype)

    device = select_device(args.device)
    if device.type == "mps" and dtype == torch.float64:
        if args.device in (None, "auto"):
            print("note: MPS doesn't support float64, falling back to CPU")
            device = torch.device("cpu")
        else:
            raise SystemExit("MPS doesn't support float64; pass --dtype float32 or --device cpu.")

    recipe = LossRecipe(name=f"E:F = {args.w_E:g}:{args.w_F:g}", w_E=args.w_E, w_F=args.w_F)

    print("=" * 64)
    print(f"  device : {device_label(device)}")
    print(f"  dtype  : {dtype}")
    print(f"  recipe : {recipe.name}  (w_E={recipe.w_E}, w_F={recipe.w_F})")
    print(f"  out    : {args.out}")
    print("=" * 64)

    frames, atomic_energies = load_ethanol(args.xyz, dtype)
    frames = frames_to(frames, device)
    elements = sorted(atomic_energies.keys())
    print(f"loaded {len(frames)} frames; elements={elements}; E0s={atomic_energies}")

    train_frames, valid_frames = split(frames, args.valid_fraction, args.seed)
    print(f"train={len(train_frames)}  valid={len(valid_frames)}\n")

    models_to_run = args.models or available_models()
    rows: List[Dict[str, object]] = []

    for name in models_to_run:
        defaults = DEFAULT_HYPERPARAMS[name]
        epochs = args.epochs if args.epochs is not None else defaults["epochs"]
        batch_size = defaults["batch_size"]
        lr = defaults["lr"]
        print(f">>> {name}  (epochs={epochs}, batch={batch_size}, lr={lr})")

        # Build once so we can report n_params before training.
        torch.manual_seed(args.seed)
        proto = build_model(
            name, elements=elements, atomic_energies=atomic_energies,
            r_max=args.r_max, frames_for_init=train_frames,
        )
        n_train = sum(p.numel() for p in proto.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in proto.parameters())

        save_path = None if args.no_save else args.out / f"{name}.pt"
        t0 = time.time()
        metrics = train_one(
            train_frames, valid_frames, recipe,
            model_factory=lambda m=proto: m,
            model_name=name,
            elements=elements,
            atomic_energies=atomic_energies,
            r_max=args.r_max,
            epochs=epochs, batch_size=batch_size, lr=lr,
            seed=args.seed,
            device=device,
            # one log line every 10% of training; just enough to see life.
            log_every=max(1, epochs // 10),
            save_path=save_path,
        )
        elapsed = time.time() - t0
        rows.append({
            "model": name,
            "n_train": n_train,
            "n_total": n_total,
            "time_s": elapsed,
            "rmse_E": metrics["rmse_E_per_atom_meV"],
            "rmse_F": metrics["rmse_F_meV_per_A"],
            "max_F":  metrics["max_F_meV_per_A"],
        })
        print()

    _print_comparison(rows, recipe)


def _print_comparison(rows: List[Dict[str, object]], recipe: LossRecipe) -> None:
    """Side-by-side: model, params, train time, val_E, val_F, max|F|."""
    header = (
        f"{'model':<12} {'#params':>10} {'time (s)':>10} "
        f"{'RMSE_E meV/atom':>18} {'RMSE_F meV/A':>16} {'max_F meV/A':>14}"
    )
    print(f"\nrecipe: {recipe.name}  (w_E={recipe.w_E}, w_F={recipe.w_F})")
    print(header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['model']:<12} "
            f"{r['n_train']:>10,} "
            f"{r['time_s']:>10.1f} "
            f"{r['rmse_E']:>18.2f} "
            f"{r['rmse_F']:>16.1f} "
            f"{r['max_F']:>14.1f}"
        )


if __name__ == "__main__":
    main()
