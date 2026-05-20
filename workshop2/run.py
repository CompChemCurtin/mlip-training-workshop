"""Workshop 2 production training wrapper around mace.cli.run_train.

Usage:
    python -m workshop2.run --config workshop2/configs/<your_config>.yaml

Why this wrapper exists at all (since mace.cli.run_train already accepts
--config <yaml>): we expand ${VAR} environment-variable references inside
the YAML before handing it to MACE. That lets one config be re-used
across users and machines — the sbatch scripts set ${MLIP_DATA} and
${MLIP_RUNS} and the YAML stays unchanged.

Anything not handled here (DDP, foundation warm-start, multi-head,
checkpoint resume, SWA, LBFGS, evaluation tables) lives upstream.

Resume an interrupted run:
    python -m workshop2.run --config <your_config>.yaml --restart_latest

Multi-GPU / multi-node: launch this module under torchrun or srun and
set 'distributed: true' in the YAML. The wrapper itself doesn't care
which launcher you use; MACE's init_distributed reads RANK/LOCAL_RANK/
WORLD_SIZE/MASTER_ADDR/MASTER_PORT from the environment.
"""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import yaml

from mace.cli.run_train import run as run_train
from mace.tools import build_default_arg_parser


def expand_env(value):
    """Recursively run os.path.expandvars on every string in a YAML tree."""
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {k: expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [expand_env(v) for v in value]
    return value


def expand_yaml_to_tempfile(path: Path) -> Path:
    """Read YAML at `path`, expand env vars, write to a tempfile, return its path."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise SystemExit(f"{path}: top-level must be a YAML mapping")
    expanded = expand_env(raw)
    fd, tmp = tempfile.mkstemp(prefix="workshop2_", suffix=".yaml")
    with os.fdopen(fd, "w") as f:
        yaml.safe_dump(expanded, f)
    return Path(tmp)


def log_gpu_visibility() -> None:
    """One line per rank: which GPUs this process can see.

    If every rank prints device_count=1, the ranks are not being spread
    across the node's GCDs -- they all land on GPU 0 and OOM while the
    others sit idle. device_count should equal the GPUs-per-node (8 on a
    Setonix MI250X node); each rank then binds to cuda:LOCAL_RANK.
    """
    import torch
    rank = os.environ.get("RANK") or os.environ.get("SLURM_PROCID", "?")
    local_rank = os.environ.get("LOCAL_RANK") or os.environ.get("SLURM_LOCALID", "?")
    visible = (os.environ.get("ROCR_VISIBLE_DEVICES")
               or os.environ.get("HIP_VISIBLE_DEVICES")
               or os.environ.get("CUDA_VISIBLE_DEVICES") or "(unset)")
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    print(f"[workshop2.run] RANK={rank} LOCAL_RANK={local_rank} "
          f"visible_devices={visible} device_count={count}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True,
                   help="Workshop 2 YAML config (with ${VAR} env-var expansion).")
    args, extras = p.parse_known_args()

    if not args.config.exists():
        raise SystemExit(f"config file not found: {args.config}")

    log_gpu_visibility()

    expanded = expand_yaml_to_tempfile(args.config)
    mace_args = build_default_arg_parser().parse_args(
        ["--config", str(expanded), *extras]
    )
    run_train(mace_args)


if __name__ == "__main__":
    main()
