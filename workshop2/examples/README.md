# Workshop 2 — worked examples (MAD)

Two end-to-end production training runs against the same dataset, one
per MLIP architecture, both using the official upstream recipes.

| Config | Architecture | Training framework | sbatch script |
|---|---|---|---|
| [`mace_mad.yaml`](mace_mad.yaml) | MACE (`ScaleShiftMACE`) | `mace.cli.run_train` via `workshop2.run` | [`scripts/sbatch_mace_mad.sh`](../../scripts/sbatch_mace_mad.sh) |
| [`pet_mad.yaml`](pet_mad.yaml)   | PET                    | `metatrain train` (CLI, directly) | [`scripts/sbatch_pet_mad.sh`](../../scripts/sbatch_pet_mad.sh) |

## Dataset

[MAD-1.0](https://archive.materialscloud.org/records/xdsbt-a3r17):
95,595 structures across 85 elements, PBESol DFT. Used by Mazitov *et al.*
to train PET-MAD. CC-BY-4.0.

```bash
bash data/download_mad.sh                         # -> data/mad/  (next to the repo)
```

Three files (~290 MB total): `mad-train.xyz` (76,476 frames),
`mad-val.xyz` (9,560), `mad-test.xyz` (9,560).

**Paths line up automatically** if you download into the repo and submit
the job from the repo: the sbatch scripts default `MLIP_DATA` to
`$SLURM_SUBMIT_DIR/data` (and `MLIP_RUNS` to `$SLURM_SUBMIT_DIR/runs`),
which the YAMLs reference as `${MLIP_DATA}` / `${MLIP_RUNS}`. To keep the
dataset somewhere shared instead, download there and export the location
before submitting:

```bash
bash data/download_mad.sh /scratch/<project>/$USER/data/mad
export MLIP_DATA=/scratch/<project>/$USER/data
sbatch scripts/sbatch_mace_mad.sh
```

## Submitting

Each `sbatch_*.sh` script scales from 1→N nodes with a single edit
(`#SBATCH --nodes=`). Default is 1 node, 8 GPUs (a single Setonix GPU
node). Multi-node uses torchrun's c10d rendezvous on the first node in
the allocation; no extra config required.

```bash
sbatch scripts/sbatch_mace_mad.sh
# or
sbatch scripts/sbatch_pet_mad.sh
```

Fill in the `EDIT ME` block in each script with your cluster's module
loads and venv activation.

## Bring your own data

Each YAML is set up so you only need to change file paths. For a single
extxyz file, split it first:

```bash
python data/split_xyz.py my_data.xyz --ratios 0.8 0.1 0.1
# -> my_data_train.xyz / my_data_val.xyz / my_data_test.xyz
```

Then point the YAML at those files. Check that the property keys
(`energy_key` / `forces_key` for MACE; `targets.energy.key` for PET)
match what's actually inside your frames (`Properties=...` line in the
extxyz header).

## What the configs are doing

Both YAMLs are commented at the line level — read them top to bottom.
Hyperparameters reflect the upstream recommendations:

- **MACE** —
  [mace-docs.readthedocs.io](https://mace-docs.readthedocs.io/en/latest/training/setting_up.html):
  E:F = 1:100 weighted loss, AdamW + EMA, ReduceLROnPlateau, SWA tail.
  Model capacity (`num_channels`, `max_L`, `correlation`) is at the
  MACE-MP-medium scale.
- **PET** —
  [metatensor.org/metatrain/.../pet](https://metatensor.org/metatrain/latest/architectures/pet.html):
  metatrain's documented defaults verbatim (`d_pet=128`, `d_node=256`,
  `num_gnn_layers=2`, `num_attention_layers=2`, MSE loss).

Neither config is tuned for absolute SOTA on MAD — they are the
straight-line "run what the docs tell you to run" baseline you'd start
from before tuning.
