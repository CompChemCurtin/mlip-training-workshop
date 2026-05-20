"""Workshop 1 package init.

We silence two unconditional UserWarnings from upstream libraries here so
the workshop output stays readable. Neither is actionable from our code:

- `metatensor.operations` decorates an internal helper with
  `@torch.jit.script`, and the TorchScript type system grumbles about the
  hint style. Their issue to fix; we just turn down the volume.
- `e3nn/o3/_wigner.py` calls `torch.load(...)` without `weights_only=`,
  and recent torch warns whenever `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` is
  in the env (which the user's setup happens to have). Cosmetic.
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
