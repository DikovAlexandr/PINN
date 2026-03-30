# PINN Research Benchmark

This repository contains code, experiments, reference data, and benchmark artifacts for research on physics-informed neural networks and related neural PDE solvers.

Vendored and modified upstream components are documented in `THIRD_PARTY.md`.

The current research workflow is centered around the `pinnacle/` workspace. It combines:

- a DeepXDE-based PINN benchmark pipeline,
- a reorganized FBPINNs wrapper,
- a vendored PINA stack for supervised and physics-informed baselines,
- exploratory VPINN code,
- committed benchmark outputs and publication figures.

The legacy `solver/` module is no longer part of `main`. It was preserved in the `archive/solver` branch so that the main branch stays focused on the current benchmark codebase.

## Current Repository Layout

- `pinnacle/benchmark.py`: baseline PINN benchmark built on the vendored `deepxde/` stack.
- `pinnacle/benchmark_fbpinns.py`: wrapper that runs FBPINNs cases and exports results in the common `runs/` format.
- `pinnacle/benchmark_pina.py`: wrapper for the vendored `pina/` framework, including both `pina_zoo` and local reference-data benchmarks.
- `pinnacle/src/`: PDE definitions, optimizers, callbacks, plotting helpers, and summary tooling used by the main benchmark pipeline.
- `pinnacle/deepxde/`: vendored and locally modified DeepXDE code.
- `pinnacle/fbpinns/`: vendored and reorganized FBPINNs code used by the benchmark wrapper.
- `pinnacle/pina/`: vendored PINA code used for supervised and PINN-style comparisons.
- `pinnacle/ref/`: reference `.dat` files used in benchmark and supervised experiments.
- `pinnacle/data/`: scripts and assets for generating synthetic coefficients and auxiliary data.
- `pinnacle/vpinn/`: experimental VPINN direction kept in the repository, but not the primary maintained benchmark path.
- `notebooks/`: exploratory notebooks and ablation studies.
- `runs/`: committed benchmark artifacts and summaries.
- `docs/`: public notes, figures, and supporting materials for the project.
- `examples/`: older lightweight examples; useful for orientation, but not the main maintained workflow.

## Recommended Environment

The supported workflow in `main` is to install dependencies from `requirements.txt` and run scripts directly. The repository is organized primarily as a research codebase rather than a polished Python package distribution.

Recommended Python version:

- Python 3.10 or 3.11

Create and activate an environment, then install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Notes:

- `torch-geometric` is required by the vendored `pina.model` package. On some platforms it may need platform-specific wheels from the PyG installation guide.
- `pinnacle/vpinn/benchmark_vpinn.py` may require extra packages beyond the main supported environment.

`requirements.txt` and `pinnacle/requirements.txt` intentionally mirror the same environment so the project can be bootstrapped either from the repository root or from the `pinnacle/` workspace.

## Main Entry Points

Run the baseline DeepXDE-style benchmark:

```bash
python pinnacle/benchmark.py --name benchmark --device cpu --method adam --iter 20000
```

Run the FBPINNs wrapper on selected cases:

```bash
python pinnacle/benchmark_fbpinns.py --name fbpinns_benchmark --device cpu --cases Burgers1D,Poisson2D_Classic,HeatMultiscale --n-steps 20000
```

Run the vendored PINA wrapper on local reference data:

```bash
python pinnacle/benchmark_pina.py --name pina_ref --suite ref --method pina_supervised --cases Burgers1D,Poisson2D_Classic,HeatMultiscale --epochs 2000
```

Run the vendored PINA benchmark on the built-in problem zoo:

```bash
python pinnacle/benchmark_pina.py --name pina_zoo --suite pina_zoo --method pina_pinn --epochs 2000
```

All benchmark wrappers save their outputs under `runs/<timestamp>-<name>-.../` in the shared experiment format used by the summary and plotting helpers.

## What Is Already In The Repository

- benchmark wrappers and reusable training utilities in `pinnacle/`,
- committed experiment outputs in `runs/`,
- ablation and exploratory notebooks in `notebooks/`,
- publication-oriented figures and public project materials in `docs/`,
- local reference datasets in `pinnacle/ref/`.

In practice, `main` already contains both code and representative experimental artifacts. The public repository is intended to expose the benchmark workflow, runnable code, and example outputs.

## Project Status

- `main` is the canonical branch for the current benchmark code.
- `archive/solver` keeps the old standalone solver history.
- `pinnacle/` is the active research workspace.
- `vpinn/` is exploratory and not yet aligned with the main benchmark pipeline.
- some notebooks are historical or exploratory and should not be treated as the single source of truth.

## Next Cleanup Targets

The repository is now oriented around the current benchmark stack, but a few cleanup tasks still remain:

- reconcile the root packaging files with the actual `pinnacle/` layout,
- refresh legacy top-level examples if they are meant to stay,
- separate "core reproducible workflow" from exploratory utilities more explicitly,
- decide which internal materials should stay private and which public artifacts belong in the open repository.
