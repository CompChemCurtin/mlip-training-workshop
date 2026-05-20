"""Split a single extxyz file into train / val / test files.

Reads every frame from the input file with ASE, shuffles deterministically,
slices into three contiguous chunks, and writes each chunk back out as
extxyz. Frame contents are preserved verbatim — keys (REF_energy /
energy / forces / stress / ...) are not touched, so the splits feed
straight into MACE, metatrain, or anything else that reads the input
file's format.

Usage:
    python data/split_xyz.py path/to/data.xyz             # -> data_train.xyz / data_val.xyz / data_test.xyz
    python data/split_xyz.py data.xyz --ratios 0.8 0.1 0.1
    python data/split_xyz.py data.xyz --out-dir splits/   # custom output directory
    python data/split_xyz.py data.xyz --seed 42

Isolated-atom frames (info["config_type"] == "IsolatedAtom"), if present,
are copied to *every* split so each one carries its own E0 references.

Key canonicalisation (--to-ref):
    The workshop configs default to the MACE convention REF_energy /
    REF_forces / REF_stress, which ASE leaves untouched in info/arrays.
    Bare keys like `energy` / `forces` get intercepted by ASE on read and
    moved onto a SinglePointCalculator instead. Pass --to-ref to rewrite
    whatever your file carries (info keys, arrays, or calculator results)
    onto the canonical REF_* keys so the splits drop straight into the
    workshop configs:

    python data/split_xyz.py data.xyz --to-ref
    python data/split_xyz.py data.xyz --to-ref --energy-key E --forces-key F
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence

import numpy as np
from ase.io import read, write


def _split_indices(n: int, ratios: Sequence[float], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_idx, val_idx, test_idx) as numpy arrays of frame indices."""
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1.0, got {ratios} (sum={sum(ratios)})")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train = int(round(n * ratios[0]))
    n_val = int(round(n * ratios[1]))
    return (
        perm[:n_train],
        perm[n_train:n_train + n_val],
        perm[n_train + n_val:],
    )


def _partition_isolated(frames):
    """Return (isolated_atom_frames, regular_frames)."""
    iso, reg = [], []
    for a in frames:
        if a.info.get("config_type") == "IsolatedAtom":
            iso.append(a)
        else:
            reg.append(a)
    return iso, reg


def _to_ref(atoms, energy_key: str, forces_key: str, stress_key: str) -> None:
    """Rewrite energy / forces / stress onto REF_* keys, in place.

    For each property we look in three places, in order: the info/arrays
    dict under the source key; the REF_* key (already canonical, leave it);
    and a SinglePointCalculator (where ASE parks bare `energy`/`forces` on
    read). Whatever we find is written to REF_energy / REF_forces /
    REF_stress and the source key is removed. The calculator is detached at
    the end so ase.io.write doesn't re-emit bare keys alongside the REF_*
    ones.
    """
    calc = atoms.calc

    # energy -> info["REF_energy"]
    if energy_key in atoms.info:
        atoms.info["REF_energy"] = float(atoms.info.pop(energy_key))
    elif "REF_energy" not in atoms.info and calc is not None:
        try:
            atoms.info["REF_energy"] = float(atoms.get_potential_energy())
        except Exception:
            pass

    # forces -> arrays["REF_forces"]
    if forces_key in atoms.arrays:
        atoms.arrays["REF_forces"] = atoms.arrays.pop(forces_key)
    elif "REF_forces" not in atoms.arrays and calc is not None:
        try:
            atoms.arrays["REF_forces"] = np.asarray(atoms.get_forces())
        except Exception:
            pass

    # stress -> info["REF_stress"] (optional; many datasets have none)
    if stress_key in atoms.info:
        atoms.info["REF_stress"] = atoms.info.pop(stress_key)
    elif stress_key in atoms.arrays:
        atoms.info["REF_stress"] = atoms.arrays.pop(stress_key)
    elif "REF_stress" not in atoms.info and calc is not None:
        try:
            atoms.info["REF_stress"] = np.asarray(atoms.get_stress())
        except Exception:
            pass

    atoms.calc = None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="Input extxyz file with multiple frames.")
    p.add_argument("--ratios", type=float, nargs=3, default=[0.8, 0.1, 0.1],
                   metavar=("TRAIN", "VAL", "TEST"),
                   help="Split ratios (must sum to 1.0). Default: 0.8 / 0.1 / 0.1.")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out-dir", type=Path, default=None,
                   help="Output directory (default: same directory as input).")
    p.add_argument("--prefix", type=str, default=None,
                   help="Output filename prefix (default: input stem). "
                        "Writes <prefix>_train.xyz, <prefix>_val.xyz, <prefix>_test.xyz.")
    p.add_argument("--to-ref", action="store_true",
                   help="Rewrite energy/forces/stress onto the canonical "
                        "REF_energy / REF_forces / REF_stress keys.")
    p.add_argument("--energy-key", type=str, default="energy",
                   help="Source energy key when --to-ref (default: energy).")
    p.add_argument("--forces-key", type=str, default="forces",
                   help="Source forces key when --to-ref (default: forces).")
    p.add_argument("--stress-key", type=str, default="stress",
                   help="Source stress key when --to-ref (default: stress).")
    args = p.parse_args()

    frames = read(str(args.input), index=":")
    iso, reg = _partition_isolated(frames)
    print(f"loaded {len(frames)} frames from {args.input}  "
          f"({len(iso)} IsolatedAtom, {len(reg)} regular)")

    if args.to_ref:
        for a in frames:
            _to_ref(a, args.energy_key, args.forces_key, args.stress_key)
        n_e = sum("REF_energy" in a.info for a in frames)
        n_f = sum("REF_forces" in a.arrays for a in frames)
        n_s = sum("REF_stress" in a.info for a in frames)
        print(f"canonicalised to REF_* keys: {n_e} energy, {n_f} forces, {n_s} stress")

    train_idx, val_idx, test_idx = _split_indices(len(reg), args.ratios, args.seed)
    splits = {
        "train": [reg[i] for i in train_idx],
        "val":   [reg[i] for i in val_idx],
        "test":  [reg[i] for i in test_idx],
    }

    out_dir = args.out_dir or args.input.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or args.input.stem
    for split_name, split_frames in splits.items():
        # Prepend isolated-atom refs to every split so each file is
        # self-contained (loaders that read E0s from IsolatedAtom frames
        # work on any of the three).
        out_path = out_dir / f"{prefix}_{split_name}.xyz"
        write(str(out_path), iso + split_frames)
        print(f"  {split_name:>5s}: {len(split_frames):6d} frames -> {out_path}")


if __name__ == "__main__":
    main()
