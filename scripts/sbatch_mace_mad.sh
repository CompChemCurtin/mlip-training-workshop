#!/bin/bash
# Multi-GPU multi-node MACE training on the MAD dataset.
#
# Scale from 1 to N nodes by editing ONE line:
#   #SBATCH --nodes=1     ->   #SBATCH --nodes=8
#
# To submit:
#   sbatch scripts/sbatch_mace_mad.sh
#
# Cluster-specific bits (module loads, partition, account, venv activation)
# live under "EDIT ME" below. Everything else should work as-is.

#SBATCH --job-name=mace-mad
#SBATCH --account=pawsey0185-gpu        # Setonix gpu partition requires the -gpu account
#SBATCH --partition=gpu
#SBATCH --nodes=1                       # <-- multi-node? bump this.
#SBATCH --ntasks-per-node=8             # ONE TASK PER GCD -- MACE reads SLURM_LOCALID
#SBATCH --gpus-per-node=8               # Setonix: 8 GCDs/node, each auto-allocates 8 cores
#SBATCH --time=02:00:00                 # size to your run; the workshop demo is short
#SBATCH --exclusive
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
mkdir -p logs

# ----- environment --------------------------------------------------------
module use /software/projects/pawsey0185/prs/modules
module load mlip-training/workshop
# ---------------------------------------------------------------------------

# ----- data + output locations --------------------------------------------
# These expand into the YAML (${MLIP_DATA}, ${MLIP_RUNS}). Default to dirs
# alongside the repo you submit from: data/download_mad.sh writes to
# data/mad by default, so submitting from the repo lines up out of the box.
# Override by exporting MLIP_DATA / MLIP_RUNS before sbatch (e.g. to point
# at a shared scratch location used by several people).
export MLIP_DATA=${MLIP_DATA:-$SLURM_SUBMIT_DIR/data}
export MLIP_RUNS=${MLIP_RUNS:-$SLURM_SUBMIT_DIR/runs}

# Fail fast with one clear message instead of N ranks of FileNotFoundError.
if [[ ! -f "$MLIP_DATA/mad/mad-train.xyz" ]]; then
    echo "ERROR: $MLIP_DATA/mad/mad-train.xyz not found." >&2
    echo "Download MAD first:  bash data/download_mad.sh \"$MLIP_DATA/mad\"" >&2
    exit 1
fi
# --------------------------------------------------------------------------

# Flush Python stdout live so per-epoch logs appear in the .out file
# instead of sitting in a buffer until the job ends.
export PYTHONUNBUFFERED=1

# Launch: one srun task per GCD, NO torchrun. MACE's default launcher is
# 'slurm', so each task reads SLURM_PROCID / SLURM_LOCALID / SLURM_NTASKS
# and binds to its own GCD; MACE sets MASTER_ADDR from the nodelist itself.
# (Wrapping this in torchrun is what made all 8 ranks land on GPU 0: the
# single srun task reported SLURM_LOCALID=0 to every forked worker.)
export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))   # avoid the fixed-33333 collision
export NCCL_SOCKET_IFNAME=hsn0                          # Setonix Slingshot NIC (multi-node)
srun python -m workshop2.run --config workshop2/examples/mace_mad.yaml
