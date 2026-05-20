"""Run a short Langevin MD trajectory using a trained workshop1 model.

Loads a checkpoint produced by `workshop1.train` (or `workshop1.finetune`),
wraps the model in an ASE Calculator, hands that to openmm-ml's
`MLPotential('ase')`, and runs a few ps of Langevin dynamics in OpenMM.

This is the *local* demo. It exercises the OpenMM `PythonForce` callback
that openmm-ml uses for ASE-Calculator-based potentials, so the same wiring
will carry over to a real MACE checkpoint via `MLPotential('mace')` on
Setonix.

Requires:
    pip install openmm openmm-ml ase

    python -m workshop1.md \\
        --checkpoint runs/metrics_pair_morse/E_F_1_100.pt \\
        --xyz data/ethanol_subset.xyz \\
        --steps 1000

The starting structure is the first non-IsolatedAtom frame in `--xyz`.
Gas-phase ethanol, no periodic box.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import ase.io
import torch  # noqa: F401  (load_calculator uses torch internally)

from workshop1.calculator import load_calculator
from workshop1.train import select_device


def first_real_frame(xyz_path: Path):
    """Return the first frame in `xyz_path` that is not an IsolatedAtom marker."""
    for atoms in ase.io.iread(str(xyz_path)):
        if atoms.info.get("config_type") != "IsolatedAtom":
            return atoms
    raise SystemExit(f"no usable frames in {xyz_path}")


def build_topology(atoms):
    """Build a minimal OpenMM Topology matching an ASE Atoms object."""
    import openmm.app as app

    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("MOL", chain)
    for symbol in atoms.get_chemical_symbols():
        element = app.Element.getBySymbol(symbol)
        topology.addAtom(symbol, element, residue)
    return topology


class _XYZReporter:
    """Minimal OpenMM reporter that appends an extxyz frame every `interval` steps.

    Writes plain xyz with PE in the comment line. Compatible with `ase.io.read`
    and small enough to keep the demo self-contained (no MDAnalysis / mdtraj).
    """

    def __init__(self, path, interval: int, symbols):
        from openmm import unit  # local import; reporter is only built if --out
        self._unit = unit
        self.path = path
        self.interval = interval
        self.symbols = list(symbols)
        # Truncate any previous run.
        self.path.write_text("")

    def describeNextReport(self, simulation):
        steps = self.interval - simulation.currentStep % self.interval
        # (steps, needPositions, needVelocities, needForces, needEnergy, wrapPositions)
        return (steps, True, False, False, True, False)

    def report(self, simulation, state):
        unit = self._unit
        positions = state.getPositions(asNumpy=True).value_in_unit(unit.angstrom)
        pe = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        with self.path.open("a") as f:
            f.write(f"{len(self.symbols)}\n")
            f.write(f'Properties=species:S:1:pos:R:3 pbc="F F F" '
                    f'step={simulation.currentStep} '
                    f'energy_kJ_per_mol={pe:.6f}\n')
            for sym, (x, y, z) in zip(self.symbols, positions):
                f.write(f"{sym:<2s} {x:18.10f} {y:18.10f} {z:18.10f}\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Checkpoint saved by workshop1.train / workshop1.finetune.")
    p.add_argument("--xyz", type=Path, default=Path("data/ethanol_subset.xyz"),
                   help="Source of initial coordinates (first non-IsolatedAtom frame is used).")
    p.add_argument("--steps", type=int, default=1000,
                   help="Number of MD steps (timestep * steps = total time).")
    p.add_argument("--timestep-fs", type=float, default=0.5,
                   help="MD timestep in femtoseconds.")
    p.add_argument("--temperature", type=float, default=300.0,
                   help="Langevin target temperature in Kelvin.")
    p.add_argument("--friction", type=float, default=1.0,
                   help="Langevin friction coefficient in 1/ps.")
    p.add_argument("--log-every", type=int, default=50,
                   help="Print a state line every N steps.")
    p.add_argument("--out", type=Path, default=None,
                   help="If given, write a trajectory.xyz to this directory.")
    p.add_argument("--device", type=str, default="auto",
                   help="auto | cuda | cuda:0 | mps | cpu  (default: auto). "
                        "Torch device for the model — OpenMM picks its own platform.")
    args = p.parse_args()
    device = select_device(args.device)

    # Lazy import so `--help` works without openmm installed.
    import openmm
    from openmm import unit
    from openmm.app import Simulation, StateDataReporter
    from openmmml import MLPotential

    # 1. Load the trained model into an ASE Calculator.
    atoms = first_real_frame(args.xyz)
    print(f"loaded initial structure: {atoms.get_chemical_formula()} ({len(atoms)} atoms)")
    calc = load_calculator(args.checkpoint, device=device)
    atoms.calc = calc

    # Sanity check: a single energy/forces call before we hand it to OpenMM.
    t0 = time.time()
    e0 = atoms.get_potential_energy()
    f0 = atoms.get_forces()
    print(f"initial E={e0:.4f} eV  max|F|={abs(f0).max():.3f} eV/A  ({1e3*(time.time()-t0):.1f} ms)")

    # 2. Build an OpenMM Topology and System driven by the ASE Calculator.
    topology = build_topology(atoms)
    potential = MLPotential("ase")
    system = potential.createSystem(topology, calculator=calc)

    integrator = openmm.LangevinMiddleIntegrator(
        args.temperature * unit.kelvin,
        args.friction / unit.picosecond,
        args.timestep_fs * unit.femtosecond,
    )

    simulation = Simulation(topology, system, integrator)
    simulation.context.setPositions(atoms.get_positions() * unit.angstrom)
    simulation.context.setVelocitiesToTemperature(args.temperature * unit.kelvin)

    # 3. Reporter -> stdout. step / time / PE / KE / Total / T.
    simulation.reporters.append(StateDataReporter(
        sys.stdout, args.log_every,
        step=True, time=True,
        potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
        temperature=True, speed=True,
    ))

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        traj_path = args.out / "trajectory.xyz"
        symbols = atoms.get_chemical_symbols()
        simulation.reporters.append(_XYZReporter(traj_path, args.log_every, symbols))
        print(f"writing trajectory snapshots to {traj_path}")

    # 4. Run.
    print(f"running {args.steps} steps of {args.timestep_fs} fs "
          f"= {args.steps * args.timestep_fs / 1000:.2f} ps at {args.temperature} K")
    t0 = time.time()
    simulation.step(args.steps)
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
