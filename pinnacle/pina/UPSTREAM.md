# Upstream Provenance

This directory vendors PINA for use as a comparison framework inside the local benchmark workflow.

Upstream repository:

- https://github.com/mathLab/PINA

Project page:

- https://mathlab.github.io/PINA/

Related paper:

- Coscia, Ivagnes, Demo, Rozza,
  "PINA: Physics-Informed Neural networks for Advanced modeling"
- Journal of Open Source Software, 2023

Declared upstream license:

- MIT

Local role in this repository:

- used by `pinnacle/benchmark_pina.py`,
- provides supervised and physics-informed baselines integrated into the repository's shared `runs/` format.

Known local divergence:

- vendored locally to avoid dependency drift,
- repository-specific benchmark wrapper and result handling around the upstream package,
- compatibility adjustments may exist where needed for local experiments.

See also:

- `../../THIRD_PARTY.md`
- `../../licenses/PINA-MIT.txt`
