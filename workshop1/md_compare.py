"""Compare the *dynamics* under different trained models, side by side.

`train.py` and `compare_models.py` end at a static error table. That table
doesn't tell you whether the resulting potential gives the right *PES* —
only whether it gets close to the right energy and forces at the validation
frames. The real test is dynamics: take each checkpoint, run a short MD
trajectory, and compare *distributions of internal coordinates* against
the same distributions in the training data.

We compute four observables on each trajectory and on the training data:

    - C-C bond length        (bedrock; converges in <1 ps)
    - C-O bond length        (the polar bond)
    - C-C-O angle            (sensitive to angular awareness)
    - C-C-O-H dihedral       (the slow mode; OH rotation barrier)

The plot has two rows. The top row overlays the *distributions* P(q) as
smooth kernel-density curves. The bottom row turns each distribution into
the *energy profile* it implies — the potential of mean force

    W(q) = -kT ln[P(q) / J(q)]

where J(q) is the geometric volume element (r^2 for a bond, sin(theta)
for the angle, 1 for the dihedral). The histogram is *not* the potential:
it is the Boltzmann-sampled distribution, and W(q) is what you recover by
inverting it. Peaks in P(q) become minima of W(q); the OH torsion's
~1-1.5 kcal/mol barrier shows up directly as the bump in the dihedral
PMF (and kT ~ 1.0 kcal/mol at 500 K, so the wells stay populated).

A "right-ish" potential gives distributions whose means line up with the
training data and an energy profile with the same well positions and
barrier heights. A model with no angles (`pair_morse`) misshapes the
angle PMF; a model with a wrong torsion barrier gets the dihedral well
depths wrong.

    python -m workshop1.md_compare \\
        --checkpoint runs/compare/pair_morse.pt --label pair_morse \\
        --checkpoint runs/compare/bonded_ff.pt  --label bonded_ff \\
        --checkpoint runs/compare/mace.pt       --label mace \\
        --steps 4000 --temperature 500

For harmonic frequencies — the model's predicted IR-active modes at the
energy minimum — see `workshop1.vibrations`, which is a Hessian via
finite differences with the same calculator wrapper.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List

import ase.io
import numpy as np
import torch  # noqa: F401  (load_calculator uses it)

from workshop1.calculator import load_calculator
from workshop1.md import build_topology, first_real_frame
from workshop1.train import select_device


# ---------- ethanol identification --------------------------------------

# Canonical rMD17 ethanol atom ordering (verified against
# data/ethanol_subset.xyz). C_alpha is the alcohol carbon (bonded to O);
# C_beta is the methyl carbon; H8 is the hydroxyl H.
ETHANOL_INDICES = dict(
    C_alpha=0, C_beta=1, O=2,
    H_alpha=(3, 4),
    H_beta=(5, 6, 7),
    H_OH=8,
)


def assert_rmd17_ethanol(atoms) -> None:
    """Sanity check that `atoms` matches the rMD17 ethanol ordering we hard-code."""
    expected_symbols = ["C", "C", "O", "H", "H", "H", "H", "H", "H"]
    got = atoms.get_chemical_symbols()
    if got != expected_symbols:
        raise SystemExit(
            f"this script assumes the canonical rMD17 ethanol ordering "
            f"{expected_symbols}; got {got}. Adapt ETHANOL_INDICES to your dataset."
        )


# ---------- observable extraction ----------------------------------------

def _bond(positions: np.ndarray, i: int, j: int) -> float:
    """Distance |r_i - r_j| in Angstrom."""
    return float(np.linalg.norm(positions[i] - positions[j]))


def _angle(positions: np.ndarray, i: int, j: int, k: int) -> float:
    """Angle in degrees at atom j between i-j and k-j."""
    u = positions[i] - positions[j]
    v = positions[k] - positions[j]
    cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _dihedral(positions: np.ndarray, i: int, j: int, k: int, l: int) -> float:
    """Dihedral angle in degrees for atoms i-j-k-l (Praxeolitic convention)."""
    b1 = positions[j] - positions[i]
    b2 = positions[k] - positions[j]
    b3 = positions[l] - positions[k]
    b2_n = b2 / np.linalg.norm(b2)
    v = b1 - np.dot(b1, b2_n) * b2_n
    w = b3 - np.dot(b3, b2_n) * b2_n
    x = np.dot(v, w)
    y = np.dot(np.cross(b2_n, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def observables(positions: np.ndarray) -> Dict[str, float]:
    """Return CC, CO, ang_CCO, dih_CCOH for one (N, 3) snapshot."""
    Ca, Cb, O = ETHANOL_INDICES["C_alpha"], ETHANOL_INDICES["C_beta"], ETHANOL_INDICES["O"]
    H_oh = ETHANOL_INDICES["H_OH"]
    return {
        "CC":    _bond(positions, Ca, Cb),
        "CO":    _bond(positions, Ca, O),
        "ang_CCO":  _angle(positions, Cb, Ca, O),
        "dih_CCOH": _dihedral(positions, Cb, Ca, O, H_oh),
    }


def stack_observables(trajectory: np.ndarray) -> Dict[str, np.ndarray]:
    """For a (T, N, 3) trajectory, return per-observable (T,) arrays."""
    keys = ["CC", "CO", "ang_CCO", "dih_CCOH"]
    out: Dict[str, list] = {k: [] for k in keys}
    for t in range(trajectory.shape[0]):
        obs = observables(trajectory[t])
        for k in keys:
            out[k].append(obs[k])
    return {k: np.array(v) for k, v in out.items()}


def training_observables(xyz_path: Path) -> Dict[str, np.ndarray]:
    """Compute the same observables on every non-IsolatedAtom training frame."""
    frames = [a for a in ase.io.iread(str(xyz_path)) if a.info.get("config_type") != "IsolatedAtom"]
    assert_rmd17_ethanol(frames[0])
    traj = np.stack([a.get_positions() for a in frames])  # (T, 9, 3)
    return stack_observables(traj)


# ---------- MD driver ----------------------------------------------------

def run_md_trajectory(
    checkpoint_path: Path,
    initial_atoms,
    *,
    n_steps: int,
    sample_every: int,
    temperature: float,
    timestep_fs: float,
    friction_per_ps: float,
    device,
) -> np.ndarray:
    """Run Langevin MD; return positions sampled every `sample_every` steps.

    Returns an (n_samples, N_atoms, 3) array in Angstroms.
    """
    import openmm
    from openmm import unit
    from openmm.app import Simulation
    from openmmml import MLPotential

    calc = load_calculator(checkpoint_path, device=device)
    topology = build_topology(initial_atoms)
    system = MLPotential("ase").createSystem(topology, calculator=calc)
    integrator = openmm.LangevinMiddleIntegrator(
        temperature * unit.kelvin,
        friction_per_ps / unit.picosecond,
        timestep_fs * unit.femtosecond,
    )
    simulation = Simulation(topology, system, integrator)
    simulation.context.setPositions(initial_atoms.get_positions() * unit.angstrom)
    simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin)

    n_samples = n_steps // sample_every
    snapshots = np.empty((n_samples, len(initial_atoms), 3), dtype=np.float64)
    for s in range(n_samples):
        simulation.step(sample_every)
        state = simulation.context.getState(getPositions=True)
        snapshots[s] = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
    return snapshots


# ---------- plotting -----------------------------------------------------

# key, axis label, lo, hi (None = derive from training range), measure kind.
# `kind` selects the Jacobian J(q) used to turn P(q) into an energy profile
# and flags the dihedral as periodic so its KDE wraps at ±180°.
PANEL_SPEC = [
    ("CC",       "C–C bond (Å)",    None,    None,   "bond"),
    ("CO",       "C–O bond (Å)",    None,    None,   "bond"),
    ("ang_CCO",  "∠C–C–O (°)",      None,    None,   "angle"),
    ("dih_CCOH", "C–C–O–H (°)",     -180.0,  180.0,  "dihedral"),
]

KB_KCAL = 0.0019872041  # Boltzmann constant in kcal/mol/K


def _bandwidth(samples: np.ndarray, *, periodic: bool) -> float:
    """Silverman-rule KDE bandwidth; uses circular spread for periodic angles."""
    n = max(samples.size, 2)
    if periodic:
        ang = np.deg2rad(samples)
        R = float(np.hypot(np.cos(ang).mean(), np.sin(ang).mean()))
        spread = np.rad2deg(np.sqrt(-2.0 * np.log(max(R, 1e-8))))
    else:
        spread = float(samples.std())
    return max(1.06 * spread * n ** (-0.2), 1e-6)


def _kde(samples: np.ndarray, grid: np.ndarray, bw: float,
         *, period: float | None = None) -> np.ndarray:
    """Gaussian KDE of `samples` evaluated on `grid`.

    When `period` is set the kernel distance wraps onto that period, so the
    dihedral density is continuous across ±180° instead of leaking off the edge.
    """
    d = grid[:, None] - samples[None, :]
    if period is not None:
        d = (d + period / 2.0) % period - period / 2.0
    k = np.exp(-0.5 * (d / bw) ** 2)
    return k.sum(axis=1) / (samples.size * bw * np.sqrt(2.0 * np.pi))


def _jacobian(grid: np.ndarray, kind: str) -> np.ndarray:
    """Geometric volume element J(q): r² for a bond, sinθ for the angle, 1 else."""
    if kind == "bond":
        return grid ** 2
    if kind == "angle":
        return np.sin(np.deg2rad(grid))
    return np.ones_like(grid)


def _pmf(density: np.ndarray, grid: np.ndarray, kind: str, kT: float,
         *, floor_frac: float = 1e-2) -> np.ndarray:
    """Potential of mean force W(q) = -kT ln[P(q)/J(q)], zeroed at its minimum.

    Returns NaN where the sampled density falls below `floor_frac` of its peak,
    so undersampled tails break the line rather than diverging to ±∞.
    """
    jac = _jacobian(grid, kind)
    good = density > floor_frac * density.max()
    w = np.full_like(grid, np.nan)
    w[good] = -kT * np.log(density[good] / jac[good])
    w[good] -= np.nanmin(w[good])
    return w


def plot_compare(
    training: Dict[str, np.ndarray],
    runs: List[Dict[str, np.ndarray]],
    labels: List[str],
    out_path: Path,
    *,
    temperature: float,
) -> None:
    import matplotlib.pyplot as plt

    kT = KB_KCAL * temperature
    n = len(PANEL_SPEC)
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 5.6), sharex="col",
                             gridspec_kw=dict(height_ratios=[1.0, 1.0]))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for col, (key, xlabel, lo, hi, kind) in enumerate(PANEL_SPEC):
        ax_d, ax_w = axes[0, col], axes[1, col]
        ref = training[key]
        period = 360.0 if kind == "dihedral" else None

        if lo is None or hi is None:
            lo_use, hi_use = float(ref.min()), float(ref.max())
            # Pad a touch so model runs that explore outside the training
            # range are still visible.
            pad = 0.1 * (hi_use - lo_use)
            lo_use, hi_use = lo_use - pad, hi_use + pad
        else:
            lo_use, hi_use = lo, hi
        grid = np.linspace(lo_use, hi_use, 400)

        # One bandwidth per panel, taken from the training data, so every
        # curve in the panel is smoothed identically and stays comparable.
        bw = _bandwidth(ref, periodic=period is not None)

        ref_d = _kde(ref, grid, bw, period=period)
        ax_d.fill_between(grid, ref_d, color="0.78", alpha=0.85,
                          label="training", zorder=1)
        ax_d.plot(grid, ref_d, color="0.45", linewidth=1.0, zorder=2)
        ax_w.plot(grid, _pmf(ref_d, grid, kind, kT), color="0.45",
                  linewidth=2.2, label="training", zorder=2)

        for i, (run, lab) in enumerate(zip(runs, labels)):
            c = colors[i % len(colors)]
            dens = _kde(run[key], grid, bw, period=period)
            ax_d.plot(grid, dens, color=c, linewidth=1.7,
                      label=lab, zorder=3 + i)
            ax_w.plot(grid, _pmf(dens, grid, kind, kT), color=c,
                      linewidth=1.7, zorder=3 + i)

        ax_d.set_title(xlabel)
        ax_d.set_ylabel("density" if col == 0 else "")
        ax_d.set_ylim(bottom=0)
        ax_d.set_xlim(lo_use, hi_use)
        ax_w.set_xlabel(xlabel)
        ax_w.set_ylabel("W (kcal/mol)" if col == 0 else "")
        ax_w.set_ylim(bottom=-0.05)

    axes[0, 0].legend(loc="upper left", fontsize=8, frameon=False)
    fig.suptitle("distribution P(q)  (top)   and   energy profile "
                 f"W(q) = −kT ln[P/J]  (bottom),   T = {temperature:.0f} K",
                 fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    print(f"wrote {out_path}")


# ---------- entrypoint ---------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("-c", "--checkpoint", action="append", required=True,
                   type=Path, dest="checkpoints",
                   help="A checkpoint to compare. Repeat for each one.")
    p.add_argument("-l", "--label", action="append", default=None, dest="labels",
                   help="Human-readable label for the corresponding --checkpoint. "
                        "Repeat in the same order; defaults to the checkpoint stem.")
    p.add_argument("--xyz", type=Path, default=Path("data/ethanol_subset.xyz"))
    p.add_argument("--out", type=Path, default=Path("runs/md_compare"),
                   help="Directory for the comparison plot.")
    p.add_argument("--steps", type=int, default=4000,
                   help="Total MD steps per checkpoint (default 4000 = 2 ps at 0.5 fs).")
    p.add_argument("--sample-every", type=int, default=8,
                   help="Sample positions every N steps (default 8 = 4 fs).")
    p.add_argument("--timestep-fs", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=500.0,
                   help="Langevin temperature in K (default 500, matches rMD17 sampling).")
    p.add_argument("--friction", type=float, default=1.0,
                   help="Langevin friction in 1/ps.")
    p.add_argument("--device", type=str, default="auto",
                   help="auto | cuda | cuda:0 | mps | cpu  (default: auto).")
    args = p.parse_args()
    device = select_device(args.device)

    labels = args.labels or [c.stem for c in args.checkpoints]
    if len(labels) != len(args.checkpoints):
        raise SystemExit(
            f"got {len(args.checkpoints)} --checkpoint but {len(labels)} --label; "
            "supply the same number of each (or omit --label entirely)."
        )

    print(f"reference: {args.xyz}")
    training = training_observables(args.xyz)
    print(f"  {len(next(iter(training.values())))} training frames")
    print(f"running {args.steps} steps × {args.timestep_fs} fs at {args.temperature} K "
          f"per checkpoint ({args.steps * args.timestep_fs / 1000:.1f} ps)")

    initial = first_real_frame(args.xyz)
    runs: List[Dict[str, np.ndarray]] = []
    for ckpt, label in zip(args.checkpoints, labels):
        print(f"\n>>> {label}  ({ckpt})")
        t0 = time.time()
        traj = run_md_trajectory(
            ckpt, initial,
            n_steps=args.steps,
            sample_every=args.sample_every,
            temperature=args.temperature,
            timestep_fs=args.timestep_fs,
            friction_per_ps=args.friction,
            device=device,
        )
        elapsed = time.time() - t0
        obs = stack_observables(traj)
        print(f"  {traj.shape[0]} snapshots in {elapsed:.1f}s; "
              f"<CC>={obs['CC'].mean():.3f}±{obs['CC'].std():.3f} A")
        runs.append(obs)

    plot_compare(training, runs, labels, args.out / "observables.png",
                 temperature=args.temperature)


if __name__ == "__main__":
    main()
