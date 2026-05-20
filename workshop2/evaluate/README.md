# Workshop 2 — evaluation

Three small scripts for poking at a trained checkpoint after the
production training job finishes. All three accept either a MACE
`.model` file or a metatomic `.pt` file (PET) — the dispatch happens in
`workshop2.evaluate.calculator.load_calculator`, based on file
extension.

| Script | What it does |
|---|---|
| `ase_sanity.py`    | Held-out RMSE for one checkpoint. ASE-backed. |
| `holdout_rmse.py`  | Same evaluation across N checkpoints, side-by-side table. |
| `openmm_md.py`     | Short Langevin run via `openmm-ml`. Native `MLPotential('mace')` for `.model`; ASE-bridged `MLPotential('ase')` + `MetatomicCalculator` for `.pt`. |

## Held-out RMSE

After training a MACE model with `workshop2.run`, the checkpoint lives at
`<work_dir>/checkpoints/<name>_run-<seed>.model`. Evaluate it against
the test split:

```bash
python -m workshop2.evaluate.ase_sanity \
    --model <work_dir>/checkpoints/<name>_run-<seed>.model \
    --xyz data/mad/mad-test.xyz \
    --energy-key energy --forces-key forces
```

For workshop1-style data (with `REF_energy` / `REF_forces` keys) the
default key names are already right; for MAD or other bare-key datasets
pass `--energy-key energy --forces-key forces`.

## Compare multiple checkpoints

```bash
python -m workshop2.evaluate.holdout_rmse \
    --model <work_dir>/checkpoints/<name>_run-<seed>.model     --label mace \
    --model <work_dir>/checkpoints/<name>_run-<seed>_swa.model --label mace+swa \
    --model <pet_work_dir>/model.pt                            --label pet \
    --xyz data/mad/mad-test.xyz \
    --energy-key energy --forces-key forces
```

A typical comparison is MACE vs MACE+SWA vs PET on the same test split.

## Short MD

```bash
python -m workshop2.evaluate.openmm_md \
    --model <work_dir>/checkpoints/<name>_run-<seed>.model \
    --xyz <initial_structure>.xyz \
    --steps 1000 --timestep-fs 0.5 --temperature 300 \
    --out <work_dir>/md/
```

The script picks the right openmm-ml path automatically. The trajectory
is written as extxyz to `<out>/trajectory.xyz`, one frame every
`--log-every` steps; without `--out` it just prints the
StateDataReporter trace.
