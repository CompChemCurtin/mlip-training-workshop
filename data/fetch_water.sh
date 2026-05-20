#!/usr/bin/env bash
# Download water-cluster training data from data_CC_water.
#
#   bash data/fetch_water.sh [destination_dir]
#
# Pulls the four extxyz files supporting the paper into <destination_dir>/full/:
#   periodic_r2scan_dataset.extxyz             (~44 MB, 1099 frames, 378-atom periodic boxes)
#   periodic_r2scan_atomic_refs.extxyz         (atomic-energy refs for the r2SCAN baseline)
#   delta_r2scancc_aims_orca.extxyz            (~14 MB, 7183 cluster cutouts, delta CCSD(T) targets)
#   delta_r2scancc_atomic_refs_aims_orca.extxyz (atomic-energy refs for the delta head)
#
# The toy subset committed to the repo (data/water_subset.xyz) is built from
# the cluster file by data/make_toy_subset.py. Re-run that script if you
# want to regenerate it.

set -euo pipefail

DEST=${1:-data}
BASE=https://raw.githubusercontent.com/fast-group-cam/data_CC_water/main
mkdir -p "$DEST/full"

curl -sL -o "$DEST/full/periodic_r2scan_dataset.extxyz" \
    "$BASE/model/baseline/dataset/periodic_r2scan_dataset.extxyz"
curl -sL -o "$DEST/full/periodic_r2scan_atomic_refs.extxyz" \
    "$BASE/model/baseline/dataset/periodic_r2scan_atomic_refs.extxyz"
curl -sL -o "$DEST/full/delta_r2scancc_aims_orca.extxyz" \
    "$BASE/model/delta/dataset/delta_r2scancc_aims_orca.extxyz"
curl -sL -o "$DEST/full/delta_r2scancc_atomic_refs_aims_orca.extxyz" \
    "$BASE/model/delta/dataset/delta_r2scancc_atomic_refs_aims_orca.extxyz"

echo "Downloaded to $DEST/full/"
ls -lh "$DEST/full/"
