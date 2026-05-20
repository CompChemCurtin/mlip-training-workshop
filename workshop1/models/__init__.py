"""Model registry for Workshop 1.

Every model exposed here implements one method:

    predict(frames) -> {"energies": (B,) tensor,
                        "forces":   list of (N_i, 3) tensors}

That is the only interface `metrics_walkthrough.py` and `train.py` rely on.
How each model produces those numbers — from a hand-coded pair Morse to
a full MACE — is local to its module.

To add a new model:
    1. Create `workshop1/models/<name>.py` exposing `build_<name>(...)`
    2. Add `(<name>, build_<name>)` to MODEL_BUILDERS below
    3. Add a defaults row to DEFAULT_HYPERPARAMS

The factory takes the same kwargs regardless of model. Models that don't
need a particular kwarg simply ignore it.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

import torch.nn as nn

from workshop1.frames import Frame

from .pair_morse import build_pair_morse


# (name -> builder function). Builders that don't exist yet are imported
# lazily so an unrelated missing model doesn't break the registry.
def _lazy(module: str, attr: str):
    def _build(**kwargs):
        import importlib
        return getattr(importlib.import_module(module), attr)(**kwargs)
    return _build


MODEL_BUILDERS: Dict[str, Callable[..., nn.Module]] = {
    "pair_morse": build_pair_morse,
    "bonded_ff":  _lazy("workshop1.models.bonded_ff", "build_bonded_ff"),
    "pet":        _lazy("workshop1.models.pet",       "build_pet"),
    "mace":       _lazy("workshop1.models.mace",      "build_mace"),
}


# Per-model training defaults. Learning rates legitimately differ across
# architectures (different parameter scales and gradient magnitudes); the
# rest is identical. The toy dataset is 160 training frames of 9-atom
# ethanol, so full-batch (200 caps at the actual train-set size) is the
# right call: gradients are exact, no stochastic noise, and we amortise
# Python / kernel-launch overhead over the whole pass. With each predict()
# call now vectorising the batch dimension this fits in milliseconds.
DEFAULT_HYPERPARAMS: Dict[str, Dict[str, Any]] = {
    "pair_morse": dict(epochs=1000, lr=5e-2, batch_size=200),
    "bonded_ff":  dict(epochs=1000, lr=1e-2, batch_size=200),
    "pet":        dict(epochs=1000, lr=1e-3, batch_size=200),
    "mace":       dict(epochs=1000, lr=1e-2, batch_size=200),
}


def available_models() -> List[str]:
    return list(MODEL_BUILDERS)


def build_model(
    name: str,
    *,
    elements: Sequence[int],
    atomic_energies: Dict[int, float],
    r_max: float,
    frames_for_init: Sequence[Frame] | None = None,
    **kwargs,
) -> nn.Module:
    """Instantiate a model by name. See MODEL_BUILDERS for the choices.

    `frames_for_init` is supplied to models that need data to set internal
    state (e.g. bonded_ff infers connectivity from frame 0; mace computes
    avg_num_neighbors and per-atom energy mean/std). Models that don't
    need it ignore it.

    The returned module carries `_builder_name` and `_builder_kwargs`
    attributes so the checkpointing code can rebuild it fresh from scratch
    (necessary for PET, whose metatomic TensorMap caches don't move off
    GPU via `nn.Module.to("cpu")` and so can't be pickled).
    """
    if name not in MODEL_BUILDERS:
        raise ValueError(
            f"unknown model {name!r}; available: {sorted(MODEL_BUILDERS)}"
        )
    elements_list = list(elements)
    atomic_energies_dict = dict(atomic_energies)
    model = MODEL_BUILDERS[name](
        elements=elements_list,
        atomic_energies=atomic_energies_dict,
        r_max=float(r_max),
        frames_for_init=list(frames_for_init) if frames_for_init is not None else None,
        **kwargs,
    )
    model._builder_name = name
    # Stash everything we'd need to rebuild *except* frames_for_init, which is
    # itself not serialisable and only matters for constructor-time stats that
    # the saved state_dict already captures.
    model._builder_kwargs = {
        "elements": elements_list,
        "atomic_energies": atomic_energies_dict,
        "r_max": float(r_max),
        **kwargs,
    }
    return model
