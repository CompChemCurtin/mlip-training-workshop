"""Workshop 2 package init.

Same upstream-warning suppressions as workshop1 so the evaluate scripts
produce clean output. None of these are actionable from our code:

- `metatensor.operations` decorates an internal helper with
  `@torch.jit.script`, and TorchScript's type checker grumbles.
- e3nn and mace both call `torch.load(...)` without `weights_only=`,
  and recent torch warns if `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` is set.
- metatomic's ASE calculator tries to symmetrise a stress tensor that's
  NaN for non-periodic systems; harmless arithmetic warning.
"""

import warnings

warnings.filterwarnings(
    "ignore",
    message=r"The TorchScript type system doesn't support",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"Environment variable TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"invalid value encountered in scalar add",
    category=RuntimeWarning,
)
