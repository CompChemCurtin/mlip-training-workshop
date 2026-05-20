#!/usr/bin/env bash
# Download the MAD-1.0 dataset (95,595 structures, 85 elements, PBESol DFT)
# used to train PET-MAD-1.0. Files:
#
#   mad-train.xyz   232 MiB  76,476 structures
#   mad-val.xyz      29 MiB   9,560 structures
#   mad-test.xyz     29 MiB   9,560 structures
#
# Source: Mazitov et al., "Massive Atomic Diversity: a compact universal
# dataset for atomistic machine learning", Scientific Data (2025).
# https://archive.materialscloud.org/records/xdsbt-a3r17  (CC-BY-4.0)
#
# Usage:
#   data/download_mad.sh                  # -> data/mad/
#   data/download_mad.sh /path/to/dest    # -> /path/to/dest/

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${1:-${REPO_ROOT}/data/mad}"
mkdir -p "${DEST}"

BASE_URL="https://archive.materialscloud.org/records/xdsbt-a3r17/files"

for fname in mad-train.xyz mad-val.xyz mad-test.xyz; do
    out="${DEST}/${fname}"
    if [[ -f "${out}" ]]; then
        echo "==> ${fname} already exists ($(du -h "${out}" | cut -f1)), skipping"
        continue
    fi
    echo "==> downloading ${fname}"
    curl -fL --progress-bar -o "${out}" "${BASE_URL}/${fname}?download=1"
done

echo
echo "MAD dataset in ${DEST}:"
du -h "${DEST}"/*.xyz 2>/dev/null
