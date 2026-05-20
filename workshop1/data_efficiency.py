"""How much data does this model need?

`train.py` answers "what loss recipe?", `compare_models.py` answers "what
architecture?", and this script answers the third question every attendee
shows up with: "how many DFT frames do I need to budget for?"

We pick one architecture, one loss recipe, and one fixed validation set,
then re-train on training-set *subsets* of varying size. The val_F vs
N_train curve shows the diminishing-returns shape and where extra data
stops paying off.

    python -m workshop1.data_efficiency

The defaults (`bonded_ff` with sizes [25, 50, 100, 160]) finish the sweep
in ~3 min on CPU. Swap in `--model mace` if you have ~15 min of laptop
time — the curve will be steeper because MACE has more capacity to
exploit each additional frame.

The validation set is held constant across all subset sizes so the
comparison is apples-to-apples; the training subsets are the *first* N
frames of the shuffled training pool, so larger N strictly contains
smaller N.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
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
    p.add_argument("--model", type=str, default="bonded_ff", choices=available_models(),
                   help="Architecture to sweep (default: bonded_ff for a fast sweep).")
    p.add_argument("--sizes", type=int, nargs="+", default=[25, 50, 100, 160],
                   help="Training-set sizes to sweep over.")
    p.add_argument("--xyz", type=Path, default=Path("data/ethanol_subset.xyz"))
    p.add_argument("--out", type=Path, default=None,
                   help="Output directory (defaults to runs/data_eff_<model>).")
    p.add_argument("--w-E", type=float, default=1.0)
    p.add_argument("--w-F", type=float, default=100.0)
    p.add_argument("--r-max", type=float, default=5.0)
    p.add_argument("--valid-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--no-save", action="store_true")
    p.add_argument("--no-plot", action="store_true")
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

    out_dir = args.out or Path(f"runs/data_eff_{args.model}")
    recipe = LossRecipe(name=f"E:F = {args.w_E:g}:{args.w_F:g}", w_E=args.w_E, w_F=args.w_F)
    defaults = DEFAULT_HYPERPARAMS[args.model]
    epochs, batch_size, lr = defaults["epochs"], defaults["batch_size"], defaults["lr"]

    print("=" * 64)
    print(f"  device : {device_label(device)}")
    print(f"  model  : {args.model}  (epochs={epochs}, batch={batch_size}, lr={lr})")
    print(f"  recipe : {recipe.name}")
    print(f"  sizes  : {args.sizes}")
    print(f"  out    : {out_dir}")
    print("=" * 64)

    frames, atomic_energies = load_ethanol(args.xyz, dtype)
    frames = frames_to(frames, device)
    elements = sorted(atomic_energies.keys())
    train_pool, valid_frames = split(frames, args.valid_fraction, args.seed)
    print(f"train pool={len(train_pool)}  valid={len(valid_frames)}  elements={elements}\n")

    sizes = sorted(set(args.sizes))
    valid_sizes = [n for n in sizes if 1 <= n <= len(train_pool)]
    skipped = [n for n in sizes if n not in valid_sizes]
    if skipped:
        print(f"note: skipping sizes outside [1, {len(train_pool)}]: {skipped}\n")

    rows: List[Dict[str, object]] = []
    for n in valid_sizes:
        print(f">>> N_train = {n}")
        # Subset is the first N frames of the (already shuffled) pool, so
        # larger N strictly contains smaller N.
        train_subset = train_pool[:n]

        def factory(name=args.model, frames=train_subset, e0s=atomic_energies,
                    elems=elements, rmax=args.r_max):
            return build_model(
                name, elements=elems, atomic_energies=e0s,
                r_max=rmax, frames_for_init=frames,
            )

        save_path = None if args.no_save else out_dir / f"n{n:03d}.pt"
        t0 = time.time()
        metrics = train_one(
            train_subset, valid_frames, recipe,
            model_factory=factory,
            model_name=args.model,
            elements=elements,
            atomic_energies=atomic_energies,
            r_max=args.r_max,
            epochs=epochs, batch_size=batch_size, lr=lr,
            seed=args.seed,
            device=device,
            log_every=max(1, epochs // 5),
            save_path=save_path,
        )
        elapsed = time.time() - t0
        rows.append({
            "n": n,
            "time_s": elapsed,
            "rmse_E": metrics["rmse_E_per_atom_meV"],
            "rmse_F": metrics["rmse_F_meV_per_A"],
        })
        print()

    _print_table(rows)
    if not args.no_plot:
        _plot(rows, args.model, recipe, out_dir / "data_efficiency.png")


def _print_table(rows: List[Dict[str, object]]) -> None:
    header = (
        f"{'N_train':>8}  {'time (s)':>10}  "
        f"{'RMSE_E meV/atom':>18}  {'RMSE_F meV/A':>14}"
    )
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(
            f"{r['n']:>8}  {r['time_s']:>10.1f}  "
            f"{r['rmse_E']:>18.2f}  {r['rmse_F']:>14.1f}"
        )


def _plot(rows: List[Dict[str, object]], model_name: str,
          recipe: LossRecipe, out_path: Path) -> None:
    """Log-log val_F vs N (the headline plot) + linear val_E for context."""
    import matplotlib.pyplot as plt

    n = np.array([r["n"] for r in rows])
    rmse_E = np.array([r["rmse_E"] for r in rows])
    rmse_F = np.array([r["rmse_F"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.4))
    axes[0].loglog(n, rmse_F, "o-", color="C0")
    axes[0].set_xlabel("N_train")
    axes[0].set_ylabel("RMSE_F (meV/Å)")
    axes[0].set_title("force error vs data")
    axes[0].grid(True, which="both", alpha=0.3)

    axes[1].semilogx(n, rmse_E, "s-", color="C1")
    axes[1].set_xlabel("N_train")
    axes[1].set_ylabel("RMSE_E (meV/atom)")
    axes[1].set_title("energy error vs data")
    axes[1].grid(True, which="both", alpha=0.3)

    fig.suptitle(f"{model_name} • {recipe.name}", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
