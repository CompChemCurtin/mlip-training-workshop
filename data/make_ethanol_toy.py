"""Build the Workshop 1 toy dataset from rMD17 ethanol.

Source: Christensen & von Lilienfeld, "On the role of gradients for
machine learning of molecular energies and forces" (Mach. Learn.: Sci.
Technol. 1 045018, 2020), data on figshare:
https://figshare.com/articles/dataset/Revised_MD17_dataset_rMD17_/12672038

The rMD17 npz stores 100k frames of ethanol (9 atoms, C2H6O) with
recomputed PBE/def2-SVP energies and forces. We sample N frames at
random, convert kcal/mol -> eV, and write an extxyz that workshop1.train
can read with no key overrides.

Usage:
    python data/make_ethanol_toy.py --output data/ethanol_subset.xyz --n-frames 200

If the npz is not already cached, the script downloads it (~64 MB).
"""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import write

# 1 kcal/mol in eV. rMD17 stores energies and forces in kcal/mol[/A].
KCAL_PER_MOL_TO_EV = 0.04336410390059322

DEFAULT_URL = "https://ndownloader.figshare.com/files/62265733"
DEFAULT_CACHE = Path("/tmp/rmd17_ethanol.npz")

# Isolated-atom energies in eV at PBE/def2-SVP (rMD17 reference level).
# These are reasonable atomic-energy baselines; the dataset has fixed
# 6:2:1 H:C:O composition so the per-element LSQ would be rank-deficient
# without these refs. See workshop1/data.py for how they're consumed.
ISOLATED_ATOM_E0S_EV = {
    1: -13.587222780835477,    # H
    6: -1029.4889999855063,    # C
    8: -2041.8396277138045,    # O
}


def fetch(url: str, cache: Path) -> Path:
    if cache.exists():
        return cache
    print(f"downloading {url} -> {cache}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, cache)
    return cache


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n-frames", type=int, default=200)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--url", type=str, default=DEFAULT_URL)
    p.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    args = p.parse_args()

    npz_path = fetch(args.url, args.cache)
    d = np.load(npz_path)
    z = d["nuclear_charges"]
    coords = d["coords"]
    energies = d["energies"] * KCAL_PER_MOL_TO_EV
    forces = d["forces"] * KCAL_PER_MOL_TO_EV

    rng = np.random.default_rng(args.seed)
    n = min(args.n_frames, coords.shape[0])
    idx = sorted(rng.choice(coords.shape[0], size=n, replace=False).tolist())

    frames = []
    # Prepend one IsolatedAtom frame per element so the loader can pick
    # them up as E0 references (see workshop1/data.py).
    # We use the MACE convention REF_energy / REF_forces; ASE's extxyz
    # reader leaves these in info/arrays rather than moving them onto a
    # SinglePointCalculator (which it does for the bare 'energy' key).
    for atomic_number, e0 in ISOLATED_ATOM_E0S_EV.items():
        ref = Atoms(numbers=[atomic_number], positions=[[0.0, 0.0, 0.0]], pbc=False)
        ref.info["config_type"] = "IsolatedAtom"
        ref.info["REF_energy"] = float(e0)
        ref.arrays["REF_forces"] = np.zeros((1, 3))
        frames.append(ref)

    for i in idx:
        a = Atoms(numbers=z, positions=coords[i], pbc=False)
        a.info["REF_energy"] = float(energies[i])
        a.arrays["REF_forces"] = forces[i].astype(np.float64)
        frames.append(a)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(str(args.output), frames)
    print(f"wrote {n} frames to {args.output}  "
          f"(E range {energies[idx].min():.3f} .. {energies[idx].max():.3f} eV, "
          f"|F| max = {np.abs(forces[idx]).max():.3f} eV/A)")


if __name__ == "__main__":
    main()
