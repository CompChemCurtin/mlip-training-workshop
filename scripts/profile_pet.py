"""Break down where PET's per-batch time actually goes, with proper GPU sync.

HIP/CUDA kernel launches are asynchronous: a bare time.time() around a GPU
call measures the *launch*, not the compute, so naive timings are
misleading (and move around run to run). This script inserts
torch.cuda.synchronize() between phases so each number reflects real work.

It replicates PET's homogeneous-N predict path (build systems -> forward ->
autograd) and times the three phases separately, plus the optimizer step,
averaged over several iterations after a warmup.

    python scripts/profile_pet.py --device cuda --batch 200 --iters 10

Run it on the same hardware you train on. The phase that dominates is the
one worth optimising; everything else is noise.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Allow running as `python scripts/profile_pet.py` from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from workshop1.frames import load_ethanol, split
from workshop1.models import build_model
from workshop1.models.pet import _make_system


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--xyz", type=Path, default=Path("data/ethanol_subset.xyz"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch", type=int, default=200)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--dtype", type=str, default="float32", choices=["float32", "float64"])
    args = p.parse_args()

    dtype = torch.float64 if args.dtype == "float64" else torch.float32
    torch.set_default_dtype(dtype)
    device = torch.device(args.device)

    frames, atomic_energies = load_ethanol(args.xyz, dtype)
    frames = [
        type(fr)(
            positions=fr.positions.to(device),
            atomic_numbers=fr.atomic_numbers.to(device),
            energy=fr.energy.to(device),
            forces=fr.forces.to(device),
            n_atoms=fr.n_atoms,
        )
        for fr in frames
    ]
    train_frames, _ = split(frames, 0.2, 1234)
    elements = sorted(atomic_energies.keys())

    model = build_model(
        "pet", elements=elements, atomic_energies=atomic_energies,
        r_max=5.0, frames_for_init=train_frames,
    ).to(device)
    model.train()
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PET on {device}: {n_params:,} params, batch={args.batch}, "
          f"dtype={dtype}, {len(train_frames)} train frames")

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch = train_frames[:args.batch]

    # Accumulators (seconds).
    t_sys = t_fwd = t_bwd = t_opt = 0.0

    for it in range(args.warmup + args.iters):
        record = it >= args.warmup

        # --- phase 1: build systems (vesin already cached after iter 0) ---
        sync(device); s = time.time()
        positions = torch.stack([fr.positions for fr in batch]).detach().requires_grad_(True)
        meta = model._neighbour_meta(positions[0], batch[0].atomic_numbers)
        i_idx, j_idx = meta["i_idx"], meta["j_idx"]
        edge_vecs = (positions[:, j_idx] - positions[:, i_idx]).unsqueeze(-1)
        systems = [
            _make_system(positions[b], edge_vecs[b], meta)
            for b in range(len(batch))
        ]
        sync(device)
        if record: t_sys += time.time() - s

        # --- phase 2: forward ---
        s = time.time()
        result = model.core(systems, model._outputs)
        E_inter = result["energy"].block().values.reshape(-1)
        z_idx = torch.stack([model._z_to_idx[fr.atomic_numbers] for fr in batch])
        E_total = E_inter + model.E0[z_idx].sum(dim=-1)
        ref_E = torch.stack([fr.energy for fr in batch]).to(device, dtype)
        loss = ((E_total - ref_E) / batch[0].n_atoms).pow(2).mean()
        sync(device)
        if record: t_fwd += time.time() - s

        # --- phase 3: backward (forces grad + loss grad) ---
        s = time.time()
        grads = torch.autograd.grad(E_total.sum(), positions, create_graph=True)[0]
        # touch the force grad so the graph isn't dead-code-eliminated
        loss = loss + 0.0 * grads.pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        sync(device)
        if record: t_bwd += time.time() - s

        # --- phase 4: optimizer step ---
        s = time.time()
        opt.step()
        sync(device)
        if record: t_opt += time.time() - s

    n = args.iters
    total = t_sys + t_fwd + t_bwd + t_opt
    print(f"\nper-batch averages over {n} iters (ms):")
    print(f"  build systems : {1e3 * t_sys / n:8.2f}  ({100 * t_sys / total:4.1f}%)")
    print(f"  forward       : {1e3 * t_fwd / n:8.2f}  ({100 * t_fwd / total:4.1f}%)")
    print(f"  backward      : {1e3 * t_bwd / n:8.2f}  ({100 * t_bwd / total:4.1f}%)")
    print(f"  optimizer     : {1e3 * t_opt / n:8.2f}  ({100 * t_opt / total:4.1f}%)")
    print(f"  TOTAL / batch : {1e3 * total / n:8.2f}")
    n_batches = (len(train_frames) + args.batch - 1) // args.batch
    print(f"\n=> ~{1e3 * total / n * n_batches:.0f} ms/epoch "
          f"({n_batches} batches of {args.batch})")


if __name__ == "__main__":
    main()
