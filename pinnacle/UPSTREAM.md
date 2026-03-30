# Upstream Provenance

This directory began as an imported and locally modified snapshot of the PINNacle benchmark project.

Upstream repository:

- https://github.com/i207M/PINNacle

Related paper:

- Hao et al., "PINNacle: A Comprehensive Benchmark of Physics-Informed Neural Networks for Solving PDEs"
- arXiv: https://arxiv.org/abs/2306.08827

Declared upstream license:

- MIT

Local role in this repository:

- historical benchmark base,
- local benchmark wrappers and experiment tooling live around this imported structure,
- nested third-party components inside this tree are documented separately.

Important nested components:

- `deepxde/` -> see `deepxde/UPSTREAM.md`
- `fbpinns/` -> see `fbpinns/UPSTREAM.md`
- `pina/` -> see `pina/UPSTREAM.md`
- `vpinn/` -> see `vpinn/UPSTREAM.md`

Local modifications are not exhaustively enumerated here, but broadly include:

- benchmark integration into the repository-level `runs/` workflow,
- additional wrappers for FBPINNs and PINA,
- repository-specific plotting and summary helpers,
- local cleanup and packaging adjustments.
