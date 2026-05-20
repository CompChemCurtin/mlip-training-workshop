"""ASE Calculator wrapping a workshop1 model.

Every model in `workshop1.models` exposes the same one-method interface:

    predict([Frame, ...]) -> {"energies": (B,) tensor,
                              "forces":   list of (N_i, 3) tensors}

The workshop's models output ASE-native units (eV, eV/A) and Cartesian
positions in Angstrom, so there's no unit conversion. The calculator just
shuttles numpy <-> torch and packs a single-frame Frame.

It plugs into `openmm-ml`'s `MLPotential('ase')` for the OpenMM MD demo in
`workshop1.md`, but you can also drive it from any ASE workflow:

    atoms = ase.io.read("data/ethanol_subset.xyz", "0")
    atoms.calc = WorkshopCalculator(model)
    E = atoms.get_potential_energy()
    F = atoms.get_forces()
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes

from workshop1.frames import Frame


class WorkshopCalculator(Calculator):
    """ASE Calculator over any model with a `predict(frames)` method."""

    implemented_properties = ["energy", "forces"]

    def __init__(self, model: torch.nn.Module, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.model = model
        self.model.eval()
        first_param = next(self.model.parameters(), None)
        self._device = first_param.device if first_param is not None else torch.device("cpu")
        self._dtype = first_param.dtype if first_param is not None else torch.get_default_dtype()

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)

        frame = Frame(
            positions=torch.as_tensor(
                atoms.get_positions(), dtype=self._dtype, device=self._device,
            ),
            atomic_numbers=torch.as_tensor(
                atoms.numbers, dtype=torch.long, device=self._device,
            ),
            # The Frame requires energy/forces fields but the model only
            # reads positions and atomic_numbers from them. Zero placeholders.
            energy=torch.zeros((), dtype=self._dtype, device=self._device),
            forces=torch.zeros((len(atoms), 3), dtype=self._dtype, device=self._device),
            n_atoms=len(atoms),
        )

        out = self.model.predict([frame])
        energy = float(out["energies"][0].detach().cpu())
        forces = out["forces"][0].detach().cpu().numpy().astype(np.float64)

        self.results = {"energy": energy, "forces": forces}


def load_calculator(checkpoint_path, device: str | torch.device = "cpu") -> WorkshopCalculator:
    """Load a checkpoint saved by `workshop1.train` and return a ready Calculator.

    Two checkpoint formats are supported, transparent to the caller:

    1. `{"model": <live module>}` — used by models that pickle cleanly
       (pair_morse, bonded_ff, mace). The module is reloaded as-is.
    2. `{"state_dict": ..., "builder_kwargs": ...}` — used by PET, whose
       metatomic TensorMap caches can't be torch.save'd from GPU. We
       rebuild the architecture from the builder kwargs and load the
       trained weights into it.

    Sets the global torch default dtype to match the loaded model's dtype
    so downstream graph rebuilds inside `models/mace.py` get tensors of
    matching dtype. Falls back to CPU when the caller asks for MPS but the
    checkpoint is float64.
    """
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "state_dict" in ckpt:
        from workshop1.models import build_model
        model = build_model(ckpt["model_name"], frames_for_init=None,
                            **ckpt["builder_kwargs"])
        model.load_state_dict(ckpt["state_dict"])
    else:
        model = ckpt["model"]

    first_param = next(model.parameters(), None)
    target = torch.device(device) if isinstance(device, str) else device
    if (target.type == "mps"
            and first_param is not None
            and first_param.dtype == torch.float64):
        print(f"note: {checkpoint_path} is float64; MPS can't hold it, falling back to CPU.")
        target = torch.device("cpu")
    model = model.to(target)
    model.eval()
    if first_param is not None:
        torch.set_default_dtype(first_param.dtype)
    return WorkshopCalculator(model)
