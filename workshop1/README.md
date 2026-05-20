# Workshop 1 — what your loss is buying you

The toy workshop. ~2000 lines across the training entry points,
`frames.py`, and the four models in `models/`. The pedagogical point is
*not* to train the best ethanol potential. It is:

1. **The loss you write down is the model you get.** Fix the architecture,
   vary the loss recipe — `python -m workshop1.train --model mace`.
2. **The inductive bias you pick matters more than parameter count.** Fix
   the loss recipe, vary the architecture from a hand-coded pair Morse to
   a full MACE — `python -m workshop1.compare_models`.
3. **A trained potential drives dynamics.** Load a checkpoint into an
   ASE/OpenMM stack and run a short Langevin trajectory —
   `python -m workshop1.md --checkpoint <path>`.
4. **Is the PES actually right?** Two complementary checks on a trained
   checkpoint:
   - `workshop1.md_compare` — run MD under each model, compare
     internal-coordinate *distributions* against the training data
     (bond lengths, angles, dihedrals).
   - `workshop1.vibrations` — relax to the energy minimum and read off
     the harmonic frequency spectrum via a Hessian. Wildly wrong
     spectra signal a model that fits forces by accident.
5. **How much data do I need?** Re-train on subsets and watch RMSE
   converge — `python -m workshop1.data_efficiency`.

Each of those is its own short script that re-uses the same training loop
and the same `model.predict(frames)` interface.

## What MACE is, in one paragraph

MACE is an equivariant message-passing graph neural network for atoms.
Each atom is a node, each bond within a cutoff `r_max` is an edge. The
network predicts a per-atom energy contribution; total energy is the sum,
and forces and stress fall out as autograd derivatives of the energy with
respect to positions and the strain tensor. "Equivariant" means the
prediction transforms correctly under rotations of the input, which lets
the model use vector and tensor features (not just scalars) without ever
breaking that symmetry. Multi-body correlations come from a tensor product
of node features per layer that's symmetrised to a chosen body order (the
`correlation` hyperparameter). Of the four models in `workshop1/models/`,
`mace` is the only one that does this; the others are progressively
simpler strawmen.

## Quick run

The toy dataset is a 2000-frame subset of rMD17 ethanol (9 atoms, PBE/def2-SVP
energies and forces in eV / eV·Å⁻¹), already in `data/ethanol_subset.xyz`.
From the repo root, the two short demos:

```bash
# 1. Fix the architecture, vary the loss recipe (five recipes).
python -m workshop1.train --model pair_morse

# 2. Fix the loss recipe (E:F = 1:100), vary the architecture (four models).
python -m workshop1.compare_models --epochs 100   # drop --epochs for the full run
```

`pair_morse` and `bonded_ff` finish in a couple of minutes on a laptop
CPU; `pet` and `mace` are heavier (use `--epochs` to keep it short, or
just let them run). `train.py` writes one `.pt` checkpoint per recipe
under `runs/metrics_<model>/`; `compare_models.py` writes one per
architecture under `runs/compare/`. Either set feeds `workshop1.md`.

### What the recipe sweep shows

```
recipe       RMSE_E meV/atom   RMSE_F meV/A
E only            12               963     (energy good, forces bad)
F only           168               299     (forces good, energy off)
E:F = 1:1          7               300     (balanced)
E:F = 1:100        7               299     (MACE's canonical)
E:F = 100:1        8               511
```

(`bonded_ff`; illustrative numbers, the pattern is the point.) Three
things worth pausing on:

1. **"F only" still gets a reasonable per-atom energy.** Forces alone
   determine the energy surface up to an additive constant; the E0
   baselines pin the constant, so even a force-only loss gives sensible
   energies.
2. **"E only" gets large force errors.** Energy is a scalar; forces are
   3N components. There is no gradient signal for forces if they don't
   appear in the loss.
3. **The 1:100 balance is the canonical MACE recipe.** It dominates the
   force gradient while letting the energy keep the model on the right
   absolute scale.

### What `compare_models` shows

Same loss recipe (`E:F = 1:100`), four architectures. The table prints
parameter count, wall-clock, and validation RMSE side by side:

```
model         #params   approx. error
pair_morse         18   worst on both E and F (no angles, two-body only)
bonded_ff          43   good E, mid F (bonds + angles + dihedrals)
pet              ~25k   mid F, marginal gain over bonded_ff for the params
mace             ~44k   best F (equivariance by construction)
```

(Run it for live numbers — they shift with epochs, precision, and
hardware.) The reads to pull out:

- **pair_morse → bonded_ff**: a few dozen params, but angles and
  dihedrals enter the model and both metrics jump. *Angular awareness
  beats raw parameter count here.*
- **bonded_ff → pet**: orders of magnitude more parameters, modest gain
  — capacity alone isn't the story; PET's learned (augmentation-based)
  equivariance leaves some on the table.
- **pet → mace**: similar budget, equivariance built in rather than
  learned — forces come out best.

## Reading order

The modules are designed to be read top-down:

| File | What it does |
|------|--------------|
| `frames.py`            | extxyz → `Frame` dataclass; IsolatedAtom → per-element E0 baselines |
| `train.py`             | five-recipe sweep, training loop, evaluation, checkpoint saving |
| `models/__init__.py`   | model registry — common `predict(frames)` interface |
| `models/pair_morse.py` | element-pair Morse, ~18 params, the strawman |
| `models/bonded_ff.py`  | bonded force field with connectivity inferred from frame 0 |
| `models/pet.py`        | toy PET — point-cloud transformer over atoms |
| `models/mace.py`       | adapter wrapping `model.py`'s ScaleShiftMACE |
| `model.py`             | the real MACE construction (used only by `models/mace.py`) |
| `compare_models.py`    | one loss recipe, four architectures, one table |
| `data_efficiency.py`   | val error vs training-set size on a fixed model |
| `calculator.py`        | ASE Calculator wrapping any of the above |
| `md.py`                | short Langevin run via `openmm.PythonForce` |
| `md_compare.py`        | MD under each checkpoint, bond/angle/torsion histograms |
| `vibrations.py`        | harmonic frequencies via finite-difference Hessian |

### `frames.py` — what's a "frame", really

A frame in extxyz is one atomic configuration: positions, atomic numbers,
optionally a periodic cell, a total energy, and per-atom forces. We use
the MACE convention of `REF_energy` / `REF_forces` keys (with the `REF_`
prefix) so ASE's built-in reader doesn't intercept them as
`SinglePointCalculator` results.

We load the file, drop any single-atom frames marked
`config_type=IsolatedAtom` and use them as per-element energy baselines
(E0s), then split the rest into train and validation. MACE doesn't predict
raw total energies — it predicts the *interaction* part, and the total is
`Σᵢ E0[Zᵢ] + Σᵢ Eᵢ_int`. E0s come either from isolated-atom DFT
calculations (the IsolatedAtom frames here) or from a least-squares fit
to per-element counts vs. total energies, which is rank-deficient for
fixed-composition data like ours.

### `train.py` and `compare_models.py` — the loop

```
for ... in ...:                       # over recipes  (train.py)
                                      # over models   (compare_models.py)
    model = build_model(name, ...)
    optimizer = Adam(model.parameters(), lr=...)
    for epoch in range(epochs):
        for batch in train_frames:
            loss = w_E * MSE(E_pred, E_true) + w_F * MSE(F_pred, F_true)
            loss.backward()
            optimizer.step()
    save_checkpoint(out / f"{slug}.pt")
```

Both scripts use the same `train_one(...)` from `train.py`. The only
difference between them is what they iterate over (recipes vs models)
and which fields they fix. Three details worth noticing in the loop:

- **Per-atom energy normalisation.** `MSE(E_pred, E_true)` is divided by
  `n_atoms` per frame, so a 200-atom box doesn't dwarf a 9-atom one.
- **Forces are gradients.** `predict()` runs autograd through the input
  positions even in eval mode. `model.eval()` only disables
  dropout/batchnorm, not autograd.
- **MSE not RMSE during training.** We square once. RMSE is computed at
  log time after summing MSEs across the dataset.

### `models/` — four architectures, one interface

Each module exposes a single builder (`build_pair_morse`, `build_pet`, ...)
that returns an `nn.Module` with one method:

```
predict(frames) -> {"energies": (B,) tensor,
                    "forces":   list of (N_i, 3) tensors}
```

That's the only API `train.py` and `calculator.py` rely on. How each
model produces those numbers is local to its file:

- `pair_morse`: every atom pair within `r_max` contributes a learnable
  Morse term keyed by the unordered element pair. 18 parameters total.
  No angular awareness — the strawman that motivates everything else.
- `bonded_ff`: bonds + angles + torsions from connectivity inferred from
  frame 0, learnable harmonic / cosine parameters. Maps roughly onto a
  classical FF.
- `pet`: a tiny point-cloud transformer over atoms. SO(3)-equivariant by
  training data augmentation, not by construction.
- `mace`: the real one. Behind the adapter is a full `ScaleShiftMACE`
  with radial Bessel basis, spherical harmonics, message-passing layers,
  symmetric tensor-product contractions to a chosen body order, and a
  scale/shift on the learned interaction energy plus per-element E0s.

`build_model(name, ..., frames_for_init=train_frames)` is the registry's
single entry point. `frames_for_init` matters only for `mace`, which
computes `avg_num_neighbors` and the per-atom interaction-energy mean/std
from the data at construction time (they're then frozen on the model).

### Finetuning

Workshop 1 doesn't have a finetune script. The right workflow for
"start from MACE-MP and adapt to my data" is the YAML route in
Workshop 2 — set `foundation_model: small|medium|large` in
`workshop2/configs/_template.yaml` and let `mace.cli.run_train` do the
warm-start, layer-freezing, and lower-LR plumbing.

### MD: `calculator.py` + `md.py`

Every workshop model already outputs eV and eV/Å, which is what ASE
expects, so the calculator is a few lines:

```python
calc = WorkshopCalculator(model)
atoms.calc = calc
atoms.get_potential_energy()
atoms.get_forces()
```

For OpenMM we hand the calculator to openmm-ml's `MLPotential('ase')`,
which internally wraps it in an `openmm.PythonForce` — the same plumbing
that runs every other potential in openmm-ml (ANI, MACE-MP, NequIP, ...).

```bash
pip install openmm openmm-ml ase  # one-time; openmm.PythonForce is recent
python -m workshop1.md \
    --checkpoint runs/metrics_pair_morse/E_F_1_100.pt \
    --steps 1000 --timestep-fs 0.5 --temperature 300
```

The script loads the checkpoint, builds a single-residue OpenMM topology
matching the first non-IsolatedAtom frame of `--xyz`, hands it to
`MLPotential('ase').createSystem(topology, calculator=calc)`, and runs
Langevin dynamics with a `StateDataReporter` printing PE / KE / total / T.

This is the *local* demo. The same recipe runs on a real MACE checkpoint
on Setonix via `MLPotential('mace').createSystem(topology, model_path=...)`
— Workshop 2 picks that up.

### Is the PES right? (`md_compare.py`, `vibrations.py`)

Both scripts take any combination of `--checkpoint <path> --label <name>`
pairs and produce a side-by-side comparison.

`md_compare.py` runs a short Langevin trajectory under each model and
overlays four observables — C–C bond, C–O bond, ∠C–C–O, C–C–O–H dihedral
— against the same observables computed from the training-data frames.
A potential with the right PES gives histograms that line up with the
reference; one with no angular awareness will smear or shift the angle
panel; one with a wrong torsion barrier will redistribute mass across
the dihedral panel.

`vibrations.py` is the T=0 complement. For each checkpoint it relaxes
to the local energy minimum (BFGS), builds the Hessian by displacing
every atom in every direction by δ=0.01 Å and reading off forces, and
prints the resulting 21 vibrational frequencies of ethanol in a side-
by-side table. The OH stretch should land near 3650 cm⁻¹, CH stretches
around 2900–3000, CH bends and C–O / C–C stretches in the 1000–1500
band, skeletal bends and torsions below 500. Models that get bonds
right but place their bends at 2500 cm⁻¹ (looking at you, `pair_morse`)
are fitting forces without fitting the curvature.

```bash
python -m workshop1.md_compare \
    --checkpoint runs/compare/bonded_ff.pt --label bonded_ff \
    --checkpoint runs/compare/mace.pt --label mace

python -m workshop1.vibrations \
    --checkpoint runs/compare/bonded_ff.pt --label bonded_ff \
    --checkpoint runs/compare/mace.pt --label mace
```

## What you don't get (use Workshop 2)

- No multi-head training, no SWA / SWALR, no LBFGS phase.
- No virials, no dipoles, no automatic config_type weighting.
- No checkpoint resuming, no DDP, no multinode.

For all of those see Workshop 2, which wraps `mace.cli.run_train`
directly.
