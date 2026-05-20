#!/bin/bash
# Multi-GPU multi-node PET training on the MAD dataset, via metatrain.
#
# Scale from 1 to N nodes by editing ONE line:
#   #SBATCH --nodes=1     ->   #SBATCH --nodes=8
#
# To submit:
#   sbatch scripts/sbatch_pet_mad.sh
#
# Cluster-specific bits (module loads, partition, account, venv activation)
# live under "EDIT ME" below. Everything else should work as-is.

#SBATCH --job-name=pet-mad
#SBATCH --account=pawsey0185-gpu        # Setonix gpu partition requires the -gpu account
#SBATCH --partition=gpu
#SBATCH --nodes=1                       # <-- multi-node? bump this.
#SBATCH --ntasks-per-node=8             # ONE TASK PER GCD -- metatrain reads SLURM_LOCALID
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
# These expand into the YAML (${MLIP_DATA}) and the --out-dir below. Default
# to dirs alongside the repo you submit from: data/download_mad.sh writes to
# data/mad by default, so submitting from the repo lines up out of the box.
# Override by exporting MLIP_DATA / MLIP_RUNS before sbatch.
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

# Launch the metatrain CLI directly (it handles logging -> outputs/<ts>/
# train.log, the output-dir layout, and reads SLURM_PROCID/LOCALID/NTASKS
# via its own DistributedEnvironment for one rank per GCD; MASTER_ADDR from
# the nodelist, MASTER_PORT from the config's distributed_port).
export NCCL_SOCKET_IFNAME=hsn0                          # Setonix Slingshot NIC (multi-node)

# metatrain's CLI doesn't expand env vars, so expand ${MLIP_DATA} ourselves
# (once, here) into an options.yaml beside the run. The model.pt output must
# be a bare filename (metatrain copies it into its checkpoint dir), so we cd
# into the run directory and let outputs/ + model.pt land there.
mkdir -p ${MLIP_RUNS}/pet_mad
python -c 'import os,sys; sys.stdout.write(os.path.expandvars(sys.stdin.read()))' \
    < workshop2/examples/pet_mad.yaml > ${MLIP_RUNS}/pet_mad/options.yaml
cd ${MLIP_RUNS}/pet_mad
# add `--restart auto` to resume from the latest checkpoint here.
srun python -m metatrain train options.yaml -o model.pt -e extensions
