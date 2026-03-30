# Upstream Provenance

This directory vendors FBPINNs code used for local benchmark comparisons.

Upstream repository:

- https://github.com/benmoseley/FBPINNs

Related paper:

- Moseley, Nissen-Meyer, Markham,
  "Finite basis physics-informed neural networks (FBPINNs): a scalable domain decomposition approach for solving differential equations"
- Advances in Computational Mathematics, 2023

Declared upstream license:

- MIT

Local role in this repository:

- benchmark backend for `pinnacle/benchmark_fbpinns.py`,
- shared source of FBPINN and PINN-style domain-decomposition experiments.

Known local divergence:

- package reorganization into `config/`, `domain/`, `models/`, `plot/`, `problems/`, and `training/`,
- output redirection into the repository-wide `runs/` format,
- local plotting/style helpers,
- compatibility changes needed for integration with the surrounding benchmark code.

Historical note:

- the local tree reflects the legacy PyTorch-oriented benchmark workflow used in this repository,
- upstream FBPINNs has since evolved beyond that snapshot.

See also:

- `../../THIRD_PARTY.md`
- `../../licenses/FBPINNs-MIT-NOTE.md`
