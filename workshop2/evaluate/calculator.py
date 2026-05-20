"""Load a trained workshop2 model as an ASE Calculator.

The three evaluate scripts (`ase_sanity`, `openmm_md`, `holdout_rmse`)
share one job: given a path to a trained checkpoint, give me back
something I can stick on `atoms.calc = ...` and call
`get_potential_energy()` on.

The dispatch is by file extension, matching what each trainer writes:

    *.model    -> mace.calculators.MACECalculator           (mace.cli.run_train)
    *.pt       -> metatomic.torch.ase_calculator.MetatomicCalculator   (metatrain)
"""

from __future__ import annotations

from pathlib import Path


def select_device(spec: str | None) -> str:
    """Resolve a device spec to a string accepted by MACE / metatomic calculators.

    "auto" (default) prefers cuda when available, else falls back to cpu.
    We deliberately skip mps here: neither MACECalculator nor
    MetatomicCalculator officially supports it, so auto-picking it would
    just produce confusing runtime errors. Pass `--device mps` explicitly
    if you want to find out.
    """
    if spec is None or spec == "auto":
        import torch
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    return spec


def load_calculator(model_path: str | Path, *, device: str = "cpu"):
    """Return an ASE Calculator for either a MACE `.model` or a metatomic `.pt`."""
    p = Path(model_path)
    if p.suffix == ".model":
        from mace.calculators import MACECalculator
        return MACECalculator(model_paths=[str(p)], device=device)
    if p.suffix == ".pt":
        from metatomic.torch.ase_calculator import MetatomicCalculator
        return MetatomicCalculator(str(p), device=device)
    raise SystemExit(
        f"unknown checkpoint type {p.suffix!r} for {p}; "
        "expected .model (MACE) or .pt (metatomic / PET)."
    )


def is_mace(model_path: str | Path) -> bool:
    return Path(model_path).suffix == ".model"
