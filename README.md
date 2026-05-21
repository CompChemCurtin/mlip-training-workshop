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

Two steps: install the `torch` wheel that matches your hardware **first**,
then everything else. This ordering matters — torch's GPU wheels live on
PyTorch's own indexes (not PyPI), and installing them first means the
later step sees torch as already satisfied and won't pull the large
default CUDA wheel over it.

### With uv

```bash
uv venv

# 1. torch for your hardware — pick ONE:
uv pip install torch --index-url https://download.pytorch.org/whl/cpu      # CPU / no GPU
uv pip install torch --index-url https://download.pytorch.org/whl/cu124    # NVIDIA CUDA
uv pip install torch --index-url https://download.pytorch.org/whl/rocm6.3  # AMD ROCm

# 2. everything else (torch already satisfied)
uv pip install mace-torch upet openmm openmmml

# 3. AMD only — HIP platform for OpenMM (skip on CPU / NVIDIA)
uv pip install openmm-hip-6
```

### With plain pip (no uv)

```bash
python -m venv .venv && source .venv/bin/activate

# 1. torch for your hardware — pick ONE:
pip install torch --index-url https://download.pytorch.org/whl/cpu      # CPU / no GPU
pip install torch --index-url https://download.pytorch.org/whl/cu124    # NVIDIA CUDA
pip install torch --index-url https://download.pytorch.org/whl/rocm6.3  # AMD ROCm

# 2. everything else (torch already satisfied)
pip install mace-torch upet openmm openmmml

# 3. AMD only — HIP platform for OpenMM (skip on CPU / NVIDIA)
pip install openmm-hip-6
```

The five top-level packages are `torch`, `mace-torch`, `upet`, `openmm`,
and `openmmml`; the rest of the stack (ase, e3nn, matplotlib, metatensor
/ metatomic / metatrain / omegaconf / vesin / scipy) is pulled in
transitively.

For a different CUDA / ROCm version, swap the index URL — e.g. `rocm6.3`
→ `rocm6.2`, or `cu124` → `cu128` (see
<https://pytorch.org/get-started/locally/>). On clusters, do the
site-specific module load first (e.g. `module load rocm/6.3.0`) and ask
your instructor for the cluster-specific module loads and sbatch
directives.
