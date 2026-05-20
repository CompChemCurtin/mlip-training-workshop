# Workshop 2 — production training (MACE and PET)

The production layer: a YAML config + your cluster's sbatch script, on
top of the upstream training tools. MACE goes through `mace.cli.run_train`
(DDP, foundation warm-start, multi-head, resume, SWA, ...); PET goes
through `metatrain`. Both run as one SLURM task per GPU.

MACE runs via a thin `workshop2.run` wrapper whose only job is to expand
`${VAR}` references in the YAML (the MACE CLI doesn't); PET uses the
`metatrain` CLI directly with the env expansion done in the sbatch. The
sbatch scripts set `${MLIP_DATA}` / `${MLIP_RUNS}` (defaulting to `data/`
and `runs/` next to the repo you submit from), so a config stays
unchanged across users and machines.

## Smoke-test locally first

Before queueing a real run, train a tiny model for a few epochs on the
workshop1 ethanol subset to confirm the install works end to end:

```bash
# MACE — ~20s on CPU, writes runs/smoke_ethanol_mace/checkpoints/*.model
python -m workshop2.run --config workshop2/configs/smoke_ethanol_mace.yaml

# PET — ~30s on CPU, writes model.pt + outputs/ in the cwd
python -m metatrain train workshop2/configs/smoke_ethanol_pet.yaml \
    -o smoke_pet_model.pt -e extensions
```

If either fails locally, fix it before touching sbatch — these are the
same code paths the cluster jobs hit.

## BYO-data workflow

For a step-by-step guide to writing the config itself — MACE vs PET,
property keys, the E0s decision, what to tune — see
[`workshop2/configs/README.md`](configs/README.md). The summary:

1. **Stage your extxyz files on shared storage** visible to all nodes
   (the repo's `data/` if it's on scratch, or any path you point
   `${MLIP_DATA}` at). Each frame should have:
   - `info["REF_energy"]`  (eV)
   - `arrays["REF_forces"]` (eV/Å)
   - `info["REF_stress"]` (eV/Å³, optional, periodic data only)

2. **Supply per-element atomic-energy references (E0s).** Three options
   in order of cleanliness:
   - prepend single-atom `config_type=IsolatedAtom` frames carrying
     `REF_energy` to your train file (recommended)
   - hardcode the dict in the YAML: `E0s: '{1: -13.6, 8: -2044.0}'`
   - leave it unset and let MACE fit them from the train file's
     varying composition (only works when composition does vary)

3. **Copy the template and edit the placeholders:**
   ```bash
   cp workshop2/configs/_template.yaml workshop2/configs/my_run.yaml
   # edit my_run.yaml: name, train_file, hyperparameters
   ```

4. **Copy the sbatch template and fill in your cluster's specifics:**
   ```bash
   cp scripts/sbatch_template.sh scripts/sbatch_my_run.sh
   # edit: <<FILL_IN_*>> markers (account, partition, modules, scratch path)
   ```

5. **Submit:**
   ```bash
   sbatch scripts/sbatch_my_run.sh
   ```

6. **Inspect the output.** MACE writes:
   - `<work_dir>/checkpoints/<name>_run-<seed>.model` — the trained model
     (load with `mace.calculators.MACECalculator(model_paths=...)`)
   - `<work_dir>/checkpoints/<name>_compiled.model` — `torch.compile`d
     version for inference
   - `<work_dir>/results/` — error tables on train/valid/test
   - `<work_dir>/logs/` — per-epoch metrics (JSONL)

## Resuming an interrupted run

Append `--restart_latest` to the wrapper command and re-submit:

```bash
python -m workshop2.run --config workshop2/configs/my_run.yaml --restart_latest
```

MACE saves a checkpoint after every evaluation interval and
`restart_latest` picks the highest-epoch one in `<work_dir>/checkpoints/`.

## Distributed training

Both MACE and metatrain expect **one SLURM task per GPU** with
`distributed: true` in the YAML: `srun --ntasks-per-node=<gpus> python -m
...`, no `torchrun`. Each task reads its rank from the SLURM environment
(`SLURM_PROCID` / `SLURM_LOCALID` / `SLURM_NTASKS`) and binds to its own
GCD; the master address is taken from the node list. The MAD sbatch
scripts already do this — see `scripts/sbatch_mace_mad.sh`.

## Multi-head training

Multi-head models share a backbone across multiple datasets / levels of
theory. Configure via the `heads` key in the YAML (a Python dict literal
mapping head_name to `{train_file, E0s, weights}`); see `_template.yaml`
for the shape. (No worked multi-head example ships yet.)

## Foundation warm-start

Set `foundation_model: small` (or `medium`/`large`, or a local `.model`
path) in the YAML to start from MACE-MP. For multi-head fine-tuning of
a foundation model see `multiheads_finetuning: true` and the
`pseudolabel_replay` family of options upstream.

## Evaluating a trained checkpoint

Three small scripts under [`workshop2/evaluate/`](evaluate/README.md):

- `ase_sanity.py` — held-out RMSE for one checkpoint
- `holdout_rmse.py` — same evaluation across N checkpoints, side-by-side
- `openmm_md.py` — short Langevin trajectory via openmm-ml (native
  `MLPotential('mace')` for `.model`, ASE-bridged for `.pt`)

All three accept either a MACE `.model` or a metatomic `.pt` (PET).
