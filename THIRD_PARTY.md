# Third-Party Components

This repository contains a mix of original project code and vendored or adapted upstream components.

The goal of this file is to make that provenance explicit:

- which parts of the tree come from upstream research codebases,
- where those codebases came from,
- which licenses they declare,
- how they are used in this repository.

Unless a file header or a subdirectory note says otherwise, treat this repository as a mixed-license codebase:

- repository-level original material is covered by the root license,
- vendored third-party components remain under their respective upstream licenses.

Recovered license texts and license notes are stored under `licenses/`.

## Component Summary

| Component | Local path | Upstream repository | Related paper / reference | Upstream license | Status in this repository |
| --- | --- | --- | --- | --- | --- |
| PINNacle benchmark core | `pinnacle/` (especially `benchmark.py`, `trainer.py`, `src/`, `ref/`, `data/`) | https://github.com/i207M/PINNacle | Hao et al., "PINNacle: A Comprehensive Benchmark of Physics-Informed Neural Networks for Solving PDEs", arXiv:2306.08827 | MIT | Imported as the original benchmark base and then locally modified |
| DeepXDE | `pinnacle/deepxde/` | https://github.com/lululxvi/deepxde | Lu et al., "DeepXDE: A deep learning library for solving differential equations", SIAM Review 2021 | LGPL-2.1 | Vendored and locally extended |
| FBPINNs | `pinnacle/fbpinns/` | https://github.com/benmoseley/FBPINNs | Moseley et al., "Finite basis physics-informed neural networks (FBPINNs): a scalable domain decomposition approach for solving differential equations", Adv. Comput. Math. 2023 | MIT | Vendored legacy PyTorch-style code, then reorganized and integrated with shared runs |
| PINA | `pinnacle/pina/` | https://github.com/mathLab/PINA | Coscia et al., "PINA: Physics-Informed Neural networks for Advanced modeling", JOSS 2023 | MIT | Vendored for local baselines and benchmark integration |
| VPINN / hp-VPINNs-style code | `pinnacle/vpinn/` | Imported as part of the PINNacle snapshot; closely related public reference: https://github.com/ehsankharazmi/hp-VPINNs | Kharazmi et al., "hp-VPINNs: Variational Physics-Informed Neural Networks With Domain Decomposition", arXiv:2003.05385 | Related public reference declares MIT; exact local file-level ancestry is only partially reconstructed | Experimental subtree kept for research continuity |

## Notes By Component

## PINNacle

The `pinnacle/` workspace started from the public PINNacle benchmark codebase and is the main historical entry point for this repository's benchmark structure.

Nested subdirectories inside `pinnacle/` are not all governed by a single provenance story:

- `pinnacle/deepxde/`, `pinnacle/fbpinns/`, `pinnacle/pina/`, and `pinnacle/vpinn/` are documented separately below,
- the benchmark wrapper structure around them is repository-local and has diverged from the upstream snapshot.

## DeepXDE

`pinnacle/deepxde/` is a vendored DeepXDE fork used by the local PINN benchmark code.

Repository-local work around this subtree includes benchmark integration and local extensions used in experiments, such as custom geometry sampling and optimizer-related experiments.

DeepXDE is licensed upstream under LGPL-2.1. See:

- `pinnacle/deepxde/UPSTREAM.md`
- `licenses/DeepXDE-LGPL-2.1-NOTE.md`

## FBPINNs

`pinnacle/fbpinns/` is a vendored FBPINNs codebase used to run FBPINN-style baselines inside the repository's shared `runs/` workflow.

Repository-local changes include:

- reorganized package layout,
- shared plotting/style helpers,
- redirected output locations so FBPINNs runs land under the repository-level `runs/` tree,
- benchmark wrapper integration through `pinnacle/benchmark_fbpinns.py`.

See:

- `pinnacle/fbpinns/UPSTREAM.md`
- `licenses/FBPINNs-MIT-NOTE.md`

## PINA

`pinnacle/pina/` is a vendored copy of PINA used for supervised and physics-informed comparison baselines.

Repository-local integration is currently centered around `pinnacle/benchmark_pina.py` and the shared result-summary format under `runs/`.

See:

- `pinnacle/pina/UPSTREAM.md`
- `licenses/PINA-MIT.txt`

## VPINN

`pinnacle/vpinn/` is retained as an experimental subtree.

Its exact public repository lineage was not fully reconstructed during the current cleanup pass, but:

- it entered this repository through the imported PINNacle benchmark snapshot,
- it clearly implements VPINN / hp-VPINN-style methods,
- some files carry narrower file-level provenance notes,
- for example, `pinnacle/vpinn/vpinn/pdebench_err.py` explicitly states that it was modified from PDEBench.

Until a full file-by-file reconstruction is done, treat `pinnacle/vpinn/` as partially reconstructed provenance with best-effort attribution.

See:

- `pinnacle/vpinn/UPSTREAM.md`
- `licenses/VPINN-PROVENANCE-NOTE.md`

## Recommended Attribution Practice

If you reuse or publish results from this repository, the safest citation pattern is:

1. cite this repository,
2. cite the relevant upstream repository or paper for each vendored component actually used,
3. preserve the component-specific provenance notes when copying subtrees out of this repository.

## File-Level Provenance

This file tracks major vendored components only.

There may also be smaller file-level borrowings or adaptations. Those should be documented directly in file headers where practical. The existing `pinnacle/vpinn/vpinn/pdebench_err.py` note is the model to follow for such cases.
