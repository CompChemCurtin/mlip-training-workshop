"""Train a workshop1 model five times in a row, varying only the loss recipe.

This is the single training entry point for Workshop 1. Pick a model with
`--model` (pair_morse | bonded_ff | pet | mace) and the script trains five
fresh copies of it on rMD17 ethanol, one per loss recipe, then prints a
side-by-side error table. One checkpoint per recipe is saved under
`<out>/<recipe>.pt` so the MD demo (workshop1.md) can load any of them.

The point is not the model. The point is: the loss you write down is the
model you get. pair_morse is the fastest demonstration (~2 min on CPU); the
same script runs against any of the workshop's models so you can repeat the
experiment with a richer functional form.

Recipes:
    E only        w_E = 1,   w_F = 0
    F only        w_E = 0,   w_F = 1
    E:F = 1:1     w_E = 1,   w_F = 1
    E:F = 1:100   w_E = 1,   w_F = 100      (MACE's default)
    E:F = 100:1   w_E = 100, w_F = 1

For each run we report:
    RMSE / MAE / max|err|  on per-atom energy   (meV/atom)
    RMSE / MAE / max|err|  on force components  (meV/A)

Run from the repo root:

    python -m workshop1.train --model pair_morse

`--plot` saves a predicted-vs-true scatter under the output dir.
"""

from __future__ import annotations

import argparse
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch

from workshop1.frames import Frame, load_ethanol, split
from workshop1.models import (
    DEFAULT_HYPERPARAMS,
    available_models,
    build_model,
)


# ---------- device handling ---------------------------------------------

def select_device(spec: str | None) -> torch.device:
    """Resolve a device spec.

    "auto" (the default) picks cuda if available, then mps, then cpu.
    Anything else is passed straight to torch.device.
    """
    if spec is None or spec == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(spec)


def device_label(dev: torch.device) -> str:
    if dev.type == "cuda":
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        return f"cuda:{idx}  ({torch.cuda.get_device_name(idx)})"
    if dev.type == "mps":
        return "mps  (Apple Metal)"
    return "cpu"


def frames_to(frames: Sequence[Frame], device: torch.device) -> List[Frame]:
    return [
        Frame(
            positions=fr.positions.to(device),
            atomic_numbers=fr.atomic_numbers.to(device),
            energy=fr.energy.to(device),
            forces=fr.forces.to(device),
            n_atoms=fr.n_atoms,
        )
        for fr in frames
    ]


# ---------- recipe + training --------------------------------------------

@dataclass
class LossRecipe:
    name: str
    w_E: float
    w_F: float

    @property
    def slug(self) -> str:
        """Filesystem-friendly version of `name`, e.g. 'E:F = 1:100' -> 'E_F_1_100'."""
        return re.sub(r"[^A-Za-z0-9]+", "_", self.name).strip("_")


def _batch_loss(model, batch: Sequence[Frame], recipe: LossRecipe):
    """Per-atom-normalised energy MSE + per-component force MSE, weighted.

    Fast path for homogeneous-N batches: stack references and predictions
    once, compute the two MSEs with a handful of device ops. The slow path
    (per-frame loop) only triggers for mixed-N data, which is rare in
    workshop1.
    """
    out = model.predict(batch)
    energies_pred = out["energies"]
    forces_pred = out["forces"]

    dtype = batch[0].positions.dtype
    device = batch[0].positions.device
    n0 = batch[0].n_atoms

    if all(fr.n_atoms == n0 for fr in batch):
        ref_E = torch.stack([fr.energy for fr in batch]).to(device, dtype)
        ref_F = torch.stack([fr.forces for fr in batch]).to(device, dtype)
        forces_stacked = (forces_pred if isinstance(forces_pred, torch.Tensor)
                          else torch.stack(forces_pred))
        L_E = ((energies_pred - ref_E) / n0).pow(2).mean()
        L_F = (forces_stacked - ref_F).pow(2).sum() / (3 * n0 * len(batch))
        return recipe.w_E * L_E + recipe.w_F * L_F

    # Heterogeneous-N fallback.
    L_E_sum = torch.zeros((), dtype=dtype, device=device)
    L_F_sum = torch.zeros((), dtype=dtype, device=device)
    n_F = 0
    for i, fr in enumerate(batch):
        e_err = (energies_pred[i] - fr.energy) / fr.n_atoms
        L_E_sum = L_E_sum + e_err.pow(2)
        L_F_sum = L_F_sum + (forces_pred[i] - fr.forces).pow(2).sum()
        n_F += 3 * fr.n_atoms
    L_E = L_E_sum / len(batch)
    L_F = L_F_sum / max(1, n_F)
    return recipe.w_E * L_E + recipe.w_F * L_F


def train_one(
    train_frames: List[Frame],
    valid_frames: List[Frame],
    recipe: LossRecipe,
    *,
    model_factory,
    model_name: str,
    elements: Sequence[int],
    atomic_energies: Dict[int, float],
    r_max: float,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    device: torch.device,
    log_every: int,
    save_path: Path | None,
) -> Dict[str, float]:
    """Train one model with `recipe`. Save it if `save_path` is given. Return val metrics.

    `model_factory()` returns a freshly initialised model. Each recipe calls
    it once. This decouples the training loop from how the model was built
    (from scratch vs from a foundation checkpoint).
    """
    torch.manual_seed(seed)
    model = model_factory()
    model.to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if n_train == n_total:
        print(f"  model on {device.type}: {n_train:,} trainable params")
    else:
        print(f"  model on {device.type}: {n_train:,} / {n_total:,} trainable params")

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    augment_fn = getattr(model, "augment", None)
    n = len(train_frames)
    rng = np.random.default_rng(seed)
    for epoch in range(epochs):
        t0 = time.time()
        order = rng.permutation(n)
        epoch_loss = 0.0
        n_steps = 0
        for start in range(0, n, batch_size):
            batch = [train_frames[i] for i in order[start:start + batch_size]]
            if augment_fn is not None:
                batch = augment_fn(batch)
            opt.zero_grad(set_to_none=True)
            loss = _batch_loss(model, batch, recipe)
            loss.backward()
            opt.step()
            epoch_loss += loss.detach().item()
            n_steps += 1

        if (epoch + 1) % log_every == 0 or epoch == epochs - 1:
            val = evaluate(model, valid_frames)
            msg = (
                f"  [{recipe.name:>13s}] epoch {epoch:4d}/{epochs}  "
                f"loss={epoch_loss / max(1, n_steps):.4f}  "
                f"val_E={val['rmse_E_per_atom_meV']:.2f} meV/atom  "
                f"val_F={val['rmse_F_meV_per_A']:.1f} meV/A  "
                f"({time.time() - t0:.1f}s)"
            )
            print(msg, flush=True)

    metrics = evaluate(model, valid_frames)
    if save_path is not None:
        save_checkpoint(save_path, model, model_name, elements, atomic_energies,
                        r_max, recipe, metrics)
        print(f"  wrote {save_path}")
    return metrics


# ---------- checkpointing ------------------------------------------------

def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    model_name: str,
    elements: Sequence[int],
    atomic_energies: Dict[int, float],
    r_max: float,
    recipe: LossRecipe,
    metrics: Dict[str, float],
) -> None:
    """Save the full module + the metadata md.py needs to rebuild the world.

    Pickles the whole `nn.Module` (not just `state_dict`) because models in
    the registry differ in their data-derived constructor state — notably
    `mace` bakes `avg_num_neighbors` and per-atom-energy mean/std into its
    constructor. The workshop is small enough that pickling the module is
    fine; for production checkpoints, prefer state_dict + builder kwargs.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    model_cpu = model.to("cpu")

    payload = {
        "model_name": model_name,
        "elements": list(elements),
        "atomic_energies": dict(atomic_energies),
        "r_max": float(r_max),
        "recipe": {"name": recipe.name, "w_E": recipe.w_E, "w_F": recipe.w_F},
        "metrics": metrics,
    }

    # Models that pickle cleanly (pair_morse, bonded_ff, mace) save the live
    # module directly — easier on the load side. PET wraps a metatomic core
    # whose TensorMap caches don't move to CPU via nn.Module.to("cpu") and so
    # can't be torch.save'd from a GPU run. For PET specifically we save
    # state_dict + builder kwargs so the loader can rebuild fresh and load
    # the trained weights.
    builder_kwargs = getattr(model_cpu, "_builder_kwargs", None)
    if model_name == "pet" and builder_kwargs is not None:
        payload["state_dict"] = model_cpu.state_dict()
        payload["builder_kwargs"] = builder_kwargs
    else:
        payload["model"] = model_cpu

    torch.save(payload, path)


# ---------- evaluation ----------------------------------------------------

def evaluate(model, frames: Sequence[Frame]) -> Dict[str, float]:
    """Per-atom energy and force-component error metrics on `frames`."""
    was_training = model.training
    model.eval()
    out = model.predict(frames)
    e_pred = out["energies"]
    f_pred = out["forces"]

    # Build the error arrays with one device->host transfer instead of N.
    ref_E = torch.stack([fr.energy for fr in frames]).to(e_pred.device, e_pred.dtype)
    n_atoms_t = torch.tensor([fr.n_atoms for fr in frames],
                             dtype=e_pred.dtype, device=e_pred.device)
    e_err = ((e_pred - ref_E) / n_atoms_t).detach().cpu().numpy()
    f_err = np.concatenate([
        (f_pred[i] - frames[i].forces).detach().cpu().numpy().ravel()
        for i in range(len(frames))
    ])
    if was_training:
        model.train()
    return {
        "rmse_E_per_atom_meV": 1e3 * float(np.sqrt(np.mean(e_err ** 2))),
        "mae_E_per_atom_meV":  1e3 * float(np.mean(np.abs(e_err))),
        "max_E_per_atom_meV":  1e3 * float(np.max(np.abs(e_err))),
        "rmse_F_meV_per_A":    1e3 * float(np.sqrt(np.mean(f_err ** 2))),
        "mae_F_meV_per_A":     1e3 * float(np.mean(np.abs(f_err))),
        "max_F_meV_per_A":     1e3 * float(np.max(np.abs(f_err))),
    }


def print_table(results: Dict[str, Dict[str, float]]) -> None:
    cols = [
        ("rmse_E_per_atom_meV", "RMSE_E meV/atom"),
        ("mae_E_per_atom_meV",  " MAE_E meV/atom"),
        ("max_E_per_atom_meV",  " max_E meV/atom"),
        ("rmse_F_meV_per_A",    " RMSE_F meV/A"),
        ("mae_F_meV_per_A",     "  MAE_F meV/A"),
        ("max_F_meV_per_A",     "  max_F meV/A"),
    ]
    name_w = max(len(n) for n in results)
    header = f"{'recipe':<{name_w}}  " + "  ".join(f"{h:>15s}" for _, h in cols)
    print("\n" + header)
    print("-" * len(header))
    for name, row in results.items():
        cells = "  ".join(f"{row[k]:15.2f}" for k, _ in cols)
        print(f"{name:<{name_w}}  {cells}")


# ---------- scatter plot (optional) --------------------------------------

def scatter_plot(
    elements: List[int],
    atomic_energies: Dict[int, float],
    train_frames: List[Frame],
    valid_frames: List[Frame],
    recipes: List[LossRecipe],
    out_path: Path,
    *,
    model_name: str,
    epochs: int,
    batch_size: int,
    lr: float,
    r_max: float,
    seed: int,
    device: torch.device,
    model_kwargs: dict | None = None,
) -> None:
    """Retrain and plot predicted-vs-true E and F for each recipe."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, len(recipes), figsize=(3.4 * len(recipes), 6.0))
    for j, recipe in enumerate(recipes):
        torch.manual_seed(seed)
        model = build_model(
            model_name,
            elements=elements,
            atomic_energies=atomic_energies,
            r_max=r_max,
            frames_for_init=train_frames,
            **(model_kwargs or {}),
        )
        model.to(device)
        model.train()
        augment_fn = getattr(model, "augment", None)
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            order = rng.permutation(len(train_frames))
            for start in range(0, len(train_frames), batch_size):
                batch = [train_frames[i] for i in order[start:start + batch_size]]
                if augment_fn is not None:
                    batch = augment_fn(batch)
                opt.zero_grad(set_to_none=True)
                _batch_loss(model, batch, recipe).backward()
                opt.step()

        model.eval()
        out = model.predict(valid_frames)
        E_pred = np.array([float(out["energies"][i] / valid_frames[i].n_atoms) for i in range(len(valid_frames))])
        E_true = np.array([float(fr.energy / fr.n_atoms) for fr in valid_frames])
        F_pred = np.concatenate([out["forces"][i].detach().cpu().numpy().ravel() for i in range(len(valid_frames))])
        F_true = np.concatenate([fr.forces.detach().cpu().numpy().ravel() for fr in valid_frames])

        ax = axes[0, j]
        ax.scatter(E_true, E_pred, s=8, alpha=0.6)
        lim = [min(E_true.min(), E_pred.min()), max(E_true.max(), E_pred.max())]
        ax.plot(lim, lim, "k--", lw=0.7)
        ax.set_title(recipe.name, fontsize=10)
        if j == 0:
            ax.set_ylabel("E_pred (eV/atom)")
        ax.set_xlabel("E_true (eV/atom)")

        ax = axes[1, j]
        ax.scatter(F_true, F_pred, s=4, alpha=0.3)
        lim = [min(F_true.min(), F_pred.min()), max(F_true.max(), F_pred.max())]
        ax.plot(lim, lim, "k--", lw=0.7)
        if j == 0:
            ax.set_ylabel("F_pred (eV/A)")
        ax.set_xlabel("F_true (eV/A)")

    fig.suptitle(f"model: {model_name}", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"\nwrote {out_path}")


# ---------- recipes ------------------------------------------------------

def default_recipes() -> List[LossRecipe]:
    return [
        LossRecipe("E only",        w_E=1.0,   w_F=0.0),
        LossRecipe("F only",        w_E=0.0,   w_F=1.0),
        LossRecipe("E:F = 1:1",     w_E=1.0,   w_F=1.0),
        LossRecipe("E:F = 1:100",   w_E=1.0,   w_F=100.0),
        LossRecipe("E:F = 100:1",   w_E=100.0, w_F=1.0),
    ]


# ---------- entrypoint ---------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="pair_morse", choices=available_models())
    p.add_argument("--xyz", type=Path, default=Path("data/ethanol_subset.xyz"))
    p.add_argument("--out", type=Path, default=None,
                   help="Output directory (defaults to runs/metrics_<model>).")
    p.add_argument("--epochs", type=int, default=None,
                   help="Training epochs per recipe (default: model-specific).")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--r-max", type=float, default=5.0)
    p.add_argument("--valid-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--plot", action="store_true")
    p.add_argument("--no-save", action="store_true",
                   help="Skip writing checkpoints (useful for quick smoke tests).")
    p.add_argument("--device", type=str, default="auto",
                   help="auto | cuda | cuda:0 | mps | cpu  (default: auto)")
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    p.add_argument("--log-every", type=int, default=1,
                   help="Print a per-epoch log line every N epochs (default: 1).")
    args = p.parse_args()

    defaults = DEFAULT_HYPERPARAMS[args.model]
    epochs = args.epochs if args.epochs is not None else defaults["epochs"]
    batch_size = args.batch_size if args.batch_size is not None else defaults["batch_size"]
    lr = args.lr if args.lr is not None else defaults["lr"]
    out_dir = args.out or Path(f"runs/metrics_{args.model}")

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    torch.set_default_dtype(dtype)

    device = select_device(args.device)
    # MPS doesn't support float64. Auto-pick CPU instead of failing; tell
    # the user explicitly if they asked for MPS.
    if device.type == "mps" and dtype == torch.float64:
        if args.device in (None, "auto"):
            print("note: MPS doesn't support float64, falling back to CPU")
            device = torch.device("cpu")
        else:
            raise SystemExit(
                "MPS doesn't support float64. Re-run with --dtype float32 or --device cpu."
            )

    print("=" * 64)
    print(f"  device : {device_label(device)}")
    print(f"  dtype  : {dtype}")
    print(f"  model  : {args.model}")
    print(f"  epochs : {epochs}")
    print(f"  batch  : {batch_size}")
    print(f"  lr     : {lr}")
    print(f"  out    : {out_dir}")
    print("=" * 64)

    frames, atomic_energies = load_ethanol(args.xyz, dtype)
    frames = frames_to(frames, device)
    elements = sorted(atomic_energies.keys())
    print(f"loaded {len(frames)} frames; elements={elements}; E0s={atomic_energies}")

    train_frames, valid_frames = split(frames, args.valid_fraction, args.seed)
    print(f"train={len(train_frames)}  valid={len(valid_frames)}")

    recipes = default_recipes()

    def factory():
        return build_model(
            args.model,
            elements=elements,
            atomic_energies=atomic_energies,
            r_max=args.r_max,
            frames_for_init=train_frames,
        )

    results: Dict[str, Dict[str, float]] = {}
    for recipe in recipes:
        print(f"\n>>> recipe '{recipe.name}'  (w_E={recipe.w_E}, w_F={recipe.w_F})")
        save_path = None if args.no_save else out_dir / f"{recipe.slug}.pt"
        results[recipe.name] = train_one(
            train_frames, valid_frames, recipe,
            model_factory=factory,
            model_name=args.model,
            elements=elements,
            atomic_energies=atomic_energies,
            r_max=args.r_max,
            epochs=epochs, batch_size=batch_size, lr=lr,
            seed=args.seed,
            device=device, log_every=args.log_every,
            save_path=save_path,
        )

    print_table(results)

    if args.plot:
        scatter_plot(
            elements, atomic_energies, train_frames, valid_frames, recipes,
            out_dir / "metrics_scatter.png",
            model_name=args.model,
            epochs=epochs, batch_size=batch_size, lr=lr,
            r_max=args.r_max, seed=args.seed,
            device=device,
        )


if __name__ == "__main__":
    main()
