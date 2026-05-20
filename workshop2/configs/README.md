# Making a config for your own data

Five decisions, then copy a template and fill it in. Start from the
verified path: get the smoke configs running first
(`workshop2/configs/smoke_ethanol_{mace,pet}.yaml`) so you know the
wrappers work on your machine, *then* point a config at your data.

## 1. MACE or PET?

| | MACE | PET |
|---|---|---|
| template | [`_template.yaml`](_template.yaml) | [`_template_pet.yaml`](_template_pet.yaml) |
| train with | `python -m workshop2.run` (env-var wrapper) | `python -m metatrain train` (CLI directly) |
| upstream | `mace.cli.run_train` | `metatrain` |
| equivariance | built in | learned (data augmentation) |

If unsure, start with MACE — it's the more common production choice and
has the gentler config surface.

## 2. Get your data into extxyz

One frame per configuration, each carrying:
- a total energy (eV) in `info`
- per-atom forces (eV/Å) in `arrays`
- optionally a cell + stress (eV/Å³) for periodic data

The property **key names** matter — you tell the config what they are.
Two common conventions:
- `REF_energy` / `REF_forces` (the workshop convention; ASE leaves these
  in `info`/`arrays` untouched)
- `energy` / `forces` (bare keys; what MAD uses)

Peek at your file's header to see which you have:
```bash
head -2 my_data.xyz      # the Properties=... line lists the array keys
```

If your data uses bare `energy`/`forces` (or any other key names), the
splitter can rewrite them to the workshop's `REF_*` convention as it
splits — see `--to-ref` in step 3 — so you don't have to set
`energy_key`/`forces_key` in the config at all.

## 3. Split into train / val / test

```bash
python data/split_xyz.py my_data.xyz --ratios 0.8 0.1 0.1
# -> my_data_train.xyz / my_data_val.xyz / my_data_test.xyz

# bare energy/forces (or odd key names)? canonicalise to REF_* as you split:
python data/split_xyz.py my_data.xyz --to-ref
python data/split_xyz.py my_data.xyz --to-ref --energy-key E --forces-key F
```
Deterministic (`--seed`), and any `config_type=IsolatedAtom` frames are
copied into all three splits. With `--to-ref` the output uses
`REF_energy` / `REF_forces` / `REF_stress`, which the templates default
to — so you can leave the config's key settings untouched.

## 4. Atomic reference energies (E0s) — the one that bites

This is where most BYO configs go wrong. The total energy MACE/PET learns
is the *interaction* part; the per-element baseline (E0) is subtracted
first. How you supply it depends on the model and your data:

**MACE** (`E0s:` key, required when there are no IsolatedAtom frames):
- **Varying composition** (mixtures, many elements): `E0s: average`
  — least-squares-fits the baseline from the data.
- **Fixed composition** (one molecule, constant element ratios): the LSQ
  fit is rank-deficient and fails. Either
  - prepend `config_type=IsolatedAtom` frames (each a single atom with its
    DFT energy) to your train file — MACE auto-detects them, **or**
  - give an explicit dict: `E0s: '{1: -13.587, 6: -1029.489, 8: -2041.840}'`
- Leaving `E0s` unset with no IsolatedAtom frames → MACE asserts
  *"Atomic energies must be provided"* at startup.

**PET**: nothing to do. metatrain least-squares-fits a per-element
baseline internally, for any composition.

## 5. Copy, edit, launch

```bash
# MACE
cp workshop2/configs/_template.yaml my_run.yaml
# edit: name, work_dir, train/valid/test paths, energy_key/forces_key, E0s
python -m workshop2.run --config my_run.yaml

# PET (metatrain CLI directly; -o must be a bare filename, so cd into a run dir)
cp workshop2/configs/_template_pet.yaml my_pet_run.yaml   # use absolute read_from paths
mkdir -p runs/my_pet_run && cd runs/my_pet_run
python -m metatrain train $OLDPWD/my_pet_run.yaml -o model.pt -e extensions
```

What to actually change vs leave alone:
- **Change**: run name, output path, data file paths, property keys, E0s,
  `compute_stress` (true only if you have stresses).
- **Tune if you have time**: `r_max` / `cutoff`, model width
  (`num_channels` / `d_pet`), `max_num_epochs` / `num_epochs`, the loss
  weights (`forces_weight` defaults to the canonical 1:100).
- **Leave alone to start**: optimiser, scheduler, EMA/SWA, the secondary
  architecture hypers. The template defaults follow the upstream recipes.

For the cluster, copy a sbatch script (`scripts/sbatch_*_mad.sh`) and
point its `--config` at your YAML; everything else (account, partition,
module load, DDP launch) is already wired for Setonix.
