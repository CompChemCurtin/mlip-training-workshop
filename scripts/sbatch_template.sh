#!/bin/bash
# Generic SLURM launcher for Workshop 2 production training.
# Copy this file, fill in <<FILL_IN_*>> markers for your cluster, and
# submit with `sbatch scripts/<your-copy>.sh`.
#
# The launch model (used by sbatch_mace_mad.sh / sbatch_pet_mad.sh):
# ONE SLURM TASK PER GPU, no torchrun. MACE (launcher=slurm by default)
# and metatrain both read their rank from the SLURM environment
# (SLURM_PROCID / SLURM_LOCALID / SLURM_NTASKS) and bind one rank per GPU;
# the master address comes from the node list. Wrapping this in torchrun
# makes every rank inherit the same SLURM_LOCALID and pile onto GPU 0.
#
# Cluster-specific bits left as <<FILL_IN_*>>: account, partition, module
# loads / env activation, and the GPU/CPU layout.

#SBATCH --job-name=mlip
#SBATCH --account=<<FILL_IN_ACCOUNT>>
#SBATCH --partition=<<FILL_IN_PARTITION>>
#SBATCH --nodes=1                               # bump for multi-node DDP
#SBATCH --ntasks-per-node=<<FILL_IN_GPUS_PER_NODE>>   # one task per GPU
#SBATCH --gpus-per-node=<<FILL_IN_GPUS_PER_NODE>>
# Some sites (e.g. Setonix) forbid --cpus-per-task for GPU jobs and bind
# cores to GPUs automatically; add it only if your scheduler needs it.
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

# ----- 1. cluster-specific environment -----
# <<FILL_IN_ENV_SETUP>>  module loads + venv activation, e.g.:
#   module use /software/projects/<project>/<user>/modules
#   module load mlip-training/workshop

# ----- 2. data + output locations -----
# These expand into the YAML config (${MLIP_DATA}, ${MLIP_RUNS}). Default
# to dirs next to the repo you submit from.
export MLIP_DATA=${MLIP_DATA:-$SLURM_SUBMIT_DIR/data}
export MLIP_RUNS=${MLIP_RUNS:-$SLURM_SUBMIT_DIR/runs}
export PYTHONUNBUFFERED=1                        # live logs in the .out file
# For multi-node, point NCCL at the high-speed NIC, e.g.:
#   export NCCL_SOCKET_IFNAME=hsn0

CONFIG=workshop2/configs/<<FILL_IN_CONFIG>>.yaml

# ----- 3. launch: one task per GPU, no torchrun -----
# MACE (the workshop2.run wrapper expands ${VAR} then calls the CLI):
srun python -m workshop2.run --config "$CONFIG"

# PET (metatrain CLI directly; pre-expand the config, then run from the
# output dir since -o must be a bare filename). See sbatch_pet_mad.sh.

# ----- resume an interrupted run -----
# MACE: append --restart_latest to the srun line above.
# PET:  add `--restart auto` to the metatrain command.
