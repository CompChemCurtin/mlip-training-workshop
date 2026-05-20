"""Short Langevin MD driven by a trained workshop2 model.

For MACE checkpoints (`.model`) this uses openmm-ml's native
`MLPotential('mace')` path — the same code path the MACE-MP-* foundations
take, which dispatches the model through `openmm-torch`'s TorchForce.

For PET (or any other metatomic) checkpoints (`.pt`) this routes through
`MLPotential('ase')` with `metatomic.torch.ase_calculator.MetatomicCalculator`
playing the calculator role. Same `openmm.PythonForce` plumbing as the
workshop1 `md.py` demo.

    python -m workshop2.evaluate.openmm_md \\
        --model runs/smoke_ethanol_mace/checkpoints/smoke_ethanol_mace_run-1234.model \\
        --xyz data/ethanol_subset.xyz --steps 500 \\
        --out runs/smoke_ethanol_mace/md/

Initial coordinates come from the first non-IsolatedAtom frame of `--xyz`.
If `--out <dir>` is given, an extxyz trajectory is written to
`<dir>/trajectory.xyz`, one frame every `--log-every` steps.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import ase.io

from workshop2.evaluate.calculator import is_mace, load_calculator, select_device


def first_real_frame(xyz_path: Path):
    """Return the first non-IsolatedAtom frame in `xyz_path`."""
    for atoms in ase.io.iread(str(xyz_path)):
        if atoms.info.get("config_type") != "IsolatedAtom":
            return atoms
    raise SystemExit(f"no usable frames in {xyz_path}")


def build_topology(atoms):
    """Minimal single-residue OpenMM Topology matching an ASE Atoms object."""
    import openmm.app as app

    topology = app.Topology()
    chain = topology.addChain()
    residue = topology.addResidue("MOL", chain)
    for symbol in atoms.get_chemical_symbols():
        element = app.Element.getBySymbol(symbol)
        topology.addAtom(symbol, element, residue)
    return topology


class _XYZReporter:
    """Appends one extxyz frame to `path` every `interval` steps."""

    def __init__(self, path, interval, symbols):
        from openmm import unit
        self._unit = unit
        self.path = path
        self.interval = interval
        self.symbols = list(symbols)
        self.path.write_text("")

    def describeNextReport(self, simulation):
        steps = self.interval - simulation.currentStep % self.interval
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
    p.add_argument("--model", type=Path, required=True,
                   help="Path to a trained .model (MACE) or .pt (PET/metatomic).")
    p.add_argument("--xyz", type=Path, required=True,
                   help="Source of initial coordinates.")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--timestep-fs", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=300.0)
    p.add_argument("--friction", type=float, default=1.0,
                   help="Langevin friction in 1/ps.")
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out", type=Path, default=None,
                   help="Directory for trajectory.xyz (no trajectory if omitted).")
    p.add_argument("--device", type=str, default="auto",
                   help="auto | cuda | cuda:0 | cpu  (default: auto). "
                        "Drives the model; OpenMM picks its own platform.")
    args = p.parse_args()
    device = select_device(args.device)

    import openmm
    from openmm import unit
    from openmm.app import Simulation, StateDataReporter
    from openmmml import MLPotential

    atoms = first_real_frame(args.xyz)
    print(f"loaded initial structure: {atoms.get_chemical_formula()} ({len(atoms)} atoms)")

    if is_mace(args.model):
        # Native openmm-ml path: builds a TorchForce around the .model.
        # MLPotential('mace') reads `device` from the createSystem kwargs
        # via MLPotentialImpl._getTorchDevice.
        potential = MLPotential("mace", modelPath=str(args.model.resolve()))
        topology = build_topology(atoms)
        system = potential.createSystem(topology, device=device)
    else:
        # PET / metatomic: hand a MetatomicCalculator to MLPotential('ase'),
        # which wraps it in openmm.PythonForce.
        calc = load_calculator(args.model, device=device)
        atoms_for_check = atoms.copy()
        atoms_for_check.calc = calc
        e0 = atoms_for_check.get_potential_energy()
        print(f"initial E (ASE check) = {e0:.4f} eV")
        topology = build_topology(atoms)
        system = MLPotential("ase").createSystem(topology, calculator=calc)

    integrator = openmm.LangevinMiddleIntegrator(
        args.temperature * unit.kelvin,
        args.friction / unit.picosecond,
        args.timestep_fs * unit.femtosecond,
    )
    simulation = Simulation(topology, system, integrator)
    simulation.context.setPositions(atoms.get_positions() * unit.angstrom)
    simulation.context.setVelocitiesToTemperature(args.temperature * unit.kelvin)

    simulation.reporters.append(StateDataReporter(
        sys.stdout, args.log_every,
        step=True, time=True,
        potentialEnergy=True, kineticEnergy=True, totalEnergy=True,
        temperature=True, speed=True,
    ))

    if args.out is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        traj_path = args.out / "trajectory.xyz"
        simulation.reporters.append(
            _XYZReporter(traj_path, args.log_every, atoms.get_chemical_symbols())
        )
        print(f"writing trajectory snapshots to {traj_path}")

    total_ps = args.steps * args.timestep_fs / 1000
    print(f"running {args.steps} steps of {args.timestep_fs} fs "
          f"= {total_ps:.2f} ps at {args.temperature} K")
    t0 = time.time()
    simulation.step(args.steps)
    print(f"done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
