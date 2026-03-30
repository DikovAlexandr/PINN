# Upstream Provenance

This directory vendors DeepXDE code for use inside the local benchmark stack.

Upstream repository:

- https://github.com/lululxvi/deepxde

Project documentation:

- https://deepxde.readthedocs.io/

Related paper:

- Lu et al., "DeepXDE: A deep learning library for solving differential equations"
- SIAM Review, 2021

Declared upstream license:

- LGPL-2.1

Local role in this repository:

- backend library for the local PINN benchmark pipeline,
- used by `pinnacle/benchmark.py` and code under `pinnacle/src/`.

Known local divergence:

- repository-local benchmark integration,
- local geometry and sampling extensions used in experiments,
- local optimizer-related experiments and tests around the vendored stack.

See also:

- `../../THIRD_PARTY.md`
- `../../licenses/DeepXDE-LGPL-2.1-NOTE.md`
