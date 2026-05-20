"""Compare predicted harmonic frequencies across trained models.

`md_compare.py` answers "does dynamics under this potential sample the right
distributions?" — a finite-T, sampling-based question. This script answers
the complementary T=0 question: "does the curvature of the energy surface
at the minimum match between models?" That's the Hessian, and ASE's
`Vibrations` builds it from finite differences of forces.

For each checkpoint we:

    1. start from the first non-IsolatedAtom frame of the training file
    2. relax to the local minimum with BFGS (fmax = 0.01 eV/Å)
    3. compute the Hessian by displacing every atom in every direction by
       δ = 0.01 Å and reading off forces; diagonalise to get frequencies

For ethanol (9 atoms), 3N-6 = 21 vibrational modes. They are reported in
cm⁻¹, sorted, with imaginary modes printed as negative numbers (a sign the
optimisation didn't reach a real minimum on that model's PES — useful
diagnostic).

    python -m workshop1.vibrations \\
        --checkpoint runs/compare/pair_morse.pt --label pair_morse \\
        --checkpoint runs/compare/bonded_ff.pt  --label bonded_ff \\
        --checkpoint runs/compare/mace.pt       --label mace

A trained potential with the right PES should produce frequencies in
roughly the right neighbourhoods: ~3600 cm⁻¹ for the O-H stretch,
~2800-3000 for the C-H stretches, ~1500 for the H-C-H bends, ~1000 for
C-O / C-C stretches, ~300 for the C-C-O bend, low-cm⁻¹ for the OH and
methyl torsions. Use a published ethanol IR table (e.g. NIST) as the
qualitative reference.
"""

from __future__ import annotations

import argparse
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch  # noqa: F401  (load_calculator pulls torch in)

from workshop1.calculator import load_calculator
from workshop1.md import first_real_frame
from workshop1.train import select_device


def harmonic_frequencies(checkpoint: Path, initial_atoms, *,
                         device, fmax: float, delta: float) -> np.ndarray:
    """Optimise then run finite-difference vibrations. Returns frequencies in cm^-1.

    ASE returns frequencies as complex numbers — imaginary modes (where the
    Hessian has a negative eigenvalue) come back with non-zero imaginary
    part. We return real parts and tag imaginary modes by sign so the table
    can show "the OH torsion is unstable on this model" at a glance.
    """
    from ase.optimize import BFGS
    from ase.vibrations import Vibrations

    calc = load_calculator(checkpoint, device=device)
    atoms = initial_atoms.copy()
    atoms.calc = calc

    BFGS(atoms, logfile=None).run(fmax=fmax, steps=400)

    # ASE writes per-displacement cache files; isolate them in a tmpdir so
    # parallel runs over multiple checkpoints don't collide.
    tmpdir = tempfile.mkdtemp(prefix="workshop1_vib_")
    try:
        vib = Vibrations(atoms, name=os.path.join(tmpdir, "vib"), delta=delta)
        vib.run()
        freqs = vib.get_frequencies()  # complex (cm^-1)
        vib.clean()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    out = np.where(np.abs(freqs.imag) > 1e-6, -np.abs(freqs.real), freqs.real)
    return np.sort(out)


def print_table(results: Dict[str, np.ndarray]) -> None:
    """Print a side-by-side table of frequencies (sorted, one column per model)."""
    labels = list(results.keys())
    n_modes = max(len(v) for v in results.values())
    header = f"{'mode':>5}  " + "  ".join(f"{lab:>10s}" for lab in labels)
    print("\nharmonic frequencies (cm^-1; negative = imaginary mode):")
    print(header)
    print("-" * len(header))
    for i in range(n_modes):
        row = []
        for lab in labels:
            arr = results[lab]
            row.append(f"{arr[i]:>10.1f}" if i < len(arr) else f"{'-':>10s}")
        print(f"{i:>5}  " + "  ".join(row))


def plot_spectrum(results: Dict[str, np.ndarray], out_path: Path) -> None:
    """Stick spectrum: one horizontal row per model, vertical bar at each frequency."""
    import matplotlib.pyplot as plt

    labels = list(results.keys())
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(9, 1.0 + 0.7 * len(labels)))

    for i, lab in enumerate(labels):
        freqs = results[lab]
        real = freqs[freqs >= 0]
        imag = freqs[freqs < 0]
        ax.vlines(real, i - 0.35, i + 0.35,
                  color=colors[i % len(colors)], linewidth=1.4)
        if imag.size:
            ax.vlines(np.abs(imag), i - 0.35, i + 0.35,
                      color=colors[i % len(colors)], linewidth=1.4,
                      linestyle=":", alpha=0.6)

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("frequency (cm$^{-1}$)")
    # Cap at the OH-stretch region; nothing physical above ~4000 for ethanol.
    ax.set_xlim(0, 4200)
    ax.invert_yaxis()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"\nwrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", action="append", required=True,
                   type=Path, dest="checkpoints",
                   help="A checkpoint to analyse. Repeat for each one.")
    p.add_argument("--label", action="append", default=None, dest="labels",
                   help="Human-readable label for the corresponding --checkpoint.")
    p.add_argument("--xyz", type=Path, default=Path("data/ethanol_subset.xyz"))
    p.add_argument("--out", type=Path, default=Path("runs/vibrations"))
    p.add_argument("--fmax", type=float, default=0.01,
                   help="BFGS convergence force in eV/A (default 0.01).")
    p.add_argument("--delta", type=float, default=0.01,
                   help="Finite-difference displacement in A (default 0.01).")
    p.add_argument("--device", type=str, default="auto",
                   help="auto | cuda | cuda:0 | mps | cpu  (default: auto).")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip writing the stick-spectrum PNG.")
    args = p.parse_args()
    device = select_device(args.device)

    labels = args.labels or [c.stem for c in args.checkpoints]
    if len(labels) != len(args.checkpoints):
        raise SystemExit(
            f"got {len(args.checkpoints)} --checkpoint but {len(labels)} --label; "
            "supply the same number of each (or omit --label entirely)."
        )

    initial = first_real_frame(args.xyz)
    print(f"initial structure: {initial.get_chemical_formula()} ({len(initial)} atoms)")

    results: Dict[str, np.ndarray] = {}
    for ckpt, label in zip(args.checkpoints, labels):
        print(f"\n>>> {label}  ({ckpt})")
        t0 = time.time()
        freqs = harmonic_frequencies(
            ckpt, initial, device=device, fmax=args.fmax, delta=args.delta,
        )
        n_imag = int((freqs < 0).sum())
        print(f"  {len(freqs)} modes ({n_imag} imaginary)  "
              f"min={freqs.min():.1f}  max={freqs.max():.1f} cm^-1  "
              f"({time.time() - t0:.1f}s)")
        results[label] = freqs

    print_table(results)
    if not args.no_plot:
        plot_spectrum(results, args.out / "spectrum.png")


if __name__ == "__main__":
    main()
