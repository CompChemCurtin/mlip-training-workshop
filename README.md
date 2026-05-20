# MLIP Training Workshops

A two-workshop guide to training Machine Learning Interatomic Potentials,
using MACE as the running example.

## Workshop 1 — MACE from parts (laptop)

Toy training you can read end to end: four model architectures
(pair-Morse → bonded FF → PET → MACE) behind one `predict(frames)`
interface, trained on a 2000-frame rMD17 ethanol subset (included).
Each script runs in a few minutes on a laptop CPU. The three headline
ones:

- `python -m workshop1.train` — fix the architecture, vary the loss.
- `python -m workshop1.compare_models` — fix the loss, vary the
  architecture.
- `python -m workshop1.md` — drive a trained checkpoint with OpenMM.

(`data_efficiency`, `md_compare`, `vibrations` round out the set.)

→ [`workshop1/README.md`](workshop1/README.md)

## Workshop 2 — production training (supercomputer)

Train a real MACE or PET model on a multi-GPU node. Two worked examples
on the MAD-1.0 dataset (95k structures, 85 elements):
[`mace_mad.yaml`](workshop2/examples/mace_mad.yaml) via
`mace.cli.run_train`, [`pet_mad.yaml`](workshop2/examples/pet_mad.yaml)
via `metatrain`. SLURM scripts handle the multi-rank launch; bring-your-
own-data attendees copy a template and edit paths.

→ [`workshop2/README.md`](workshop2/README.md)

## Repo layout

```
workshop1/    # toy training, read every line
workshop2/    # configs + example MAD runs + evaluate scripts
data/         # dataset download + train/val/test split helpers
scripts/      # SLURM scripts (MAD examples + a generic template)
```

## Setup

```bash
uv venv && uv sync
```

Direct dependencies are just `mace-torch` and `upet`; everything else
(torch, ase, e3nn, matplotlib, the metatensor / metatomic / metatrain /
omegaconf / vesin / scipy stack) comes in transitively.

On GPU-attached clusters where `torch` needs a vendor wheel
(CUDA / ROCm), install that wheel **first** so the rest of the stack
resolves against it. Example for AMD GPUs with ROCm 6.3 (e.g. Setonix):

```bash
# site-specific module load first, e.g. `module load rocm/6.3.0`
uv venv
uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.3
uv sync
```

For other ROCm or CUDA versions, swap the index URL accordingly (see
<https://pytorch.org/get-started/locally/>). Ask your instructor for
the cluster-specific module loads and sbatch directives.
