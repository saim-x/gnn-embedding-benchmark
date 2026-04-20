# Reproduction Notes: How Embeddings Shape Graph Neural Networks: Classical vs Quantum-Oriented Node Representations

> This document records implementation choices, what the paper specifies, and what remains ambiguous.

---

## Paper

- **Title:** How Embeddings Shape Graph Neural Networks: Classical vs Quantum-Oriented Node Representations
- **Authors:** Nouhaila Innan, Antonello Rosato, Alberto Marchisio, Muhammad Shafique
- **Year:** 2026
- **ArXiv:** https://arxiv.org/abs/2604.15273v1
- **Official code:** None found

---

## What this implements

A minimal PyTorch benchmark scaffold that keeps the graph classifier fixed (GIN) and swaps embedding modules (Fixed, MLP, QuOp, QWalkVec, QPE, Angle-VQC placeholder) under one shared training/evaluation protocol. The objective is to preserve the paper’s controlled-comparison structure rather than replicate every reported table value.

---

## Verified against

- [ ] Paper equations (several equations were image-only in extracted text)
- [x] Paper Algorithm boxes (Algorithm 1/2/3)
- [ ] Official code
- [ ] Well-known reimplementation
- [x] Paper method/experiment text and hyperparameter table

---

## Unspecified choices

| Component | Our Choice | Alternatives | Paper Quote (if partial) | Section |
|---|---|---|---|---|
| QuOp hop radius `h` | `h=1` | `h=2`, `h=3` | "Require: ... hop radius h ..." | Algorithm 1 |
| QuOp qubit budget `q` | `q=5` | `q=4`, `q=6` | "Require: ... qubit budget q ..." | Algorithm 1 |
| QuOp summary vector details | Concatenate probabilities + real + imag amplitudes | Probabilities-only, amplitude moments | "Extract a fixed summary vector s_v" | Algorithm 1 |
| QWalkVec coin/shift realization | Dense transition approximation with `softmax(wp I + wq P)` | Explicit directed-edge coined walk state space | "Apply the coin and shift updates" | Algorithm 2 |
| QPE anchor selection policy | Highest-degree anchors | Random anchors, farthest-point sampling | "Select a set of anchor nodes A ⊆ V" | Algorithm 3 |
| QPE operator `H` construction | Normalized graph Laplacian | Adjacency-based operator, random-walk Laplacian | "Require: ... operator H" | Algorithm 3 |
| Adam betas/epsilon | `(0.9, 0.999), eps=1e-8` | `(0.9, 0.98), eps=1e-9` | "Adam; lr 10^-3 ..." | Table I |
| Gradient clipping | Disabled by default | Clip norm 1.0 / 5.0 | — | — |
| Angle-VQC full circuit hyperparameters | Placeholder module for interface parity | Full PennyLane circuit implementation | "Angle embedding ... followed by L_q entangling layers" | Fig. 2 caption |

---

## Known deviations

| Deviation | Paper says | We do | Reason |
|---|---|---|---|
| Angle-VQC implementation depth | Includes Angle-VQC in benchmark | Provide placeholder approximation | Full circuit details are not fully recoverable from parsed paper text in minimal mode |
| Dataset IO | Uses TU and QM9 benchmarks | Expect local `{dataset}.pt` file | This scaffold intentionally avoids auto-download and external benchmark plumbing |

---

## Expected results

Representative numbers from Table II (paper):

| Metric | Paper's number | Dataset | Conditions |
|---|---|---|---|
| Accuracy | 0.72 | IMDB (MLP) | Table II |
| Macro-F1 | 0.7172 | IMDB (MLP) | Table II |
| Accuracy | 0.9474 | MUTAG (QWalkVec*) | Table II |
| Macro-F1 | 0.9360 | MUTAG (QWalkVec*) | Table II |

Exact reproduction requires matching all unspecified choices plus data and environment.

---

## Debugging tips

1. **Unexpectedly low Macro-F1 with good accuracy:** inspect class imbalance and confusion matrix; this benchmark reports macro metrics explicitly.
2. **Unstable QuOp outputs:** reduce `quop_qubits` or add small diagonal damping before matrix exponential for numerical stability.
3. **QWalkVec underperforming in fixed mode:** verify that trainable projection is enabled when using `qwalkvec_trainable`.

---

## Scope decisions

### Implemented
- Fixed GIN backbone and shared training/evaluation protocol.
- Embedding swap interface with algorithm-inspired QuOp/QWalkVec/QPE implementations.
- Paper-reported metrics and early stopping criterion.

### Intentionally excluded
- Full baseline ecosystem reimplementations beyond what is needed to run the benchmark scaffold.
- Distributed training, experiment tracking, and dataset download tooling.
- Exact paper-figure/table recreation scripts.

### Needed for full reproduction (not included)
- Official benchmark preprocessing parity for each TU dataset and QM9 conversion pipeline.
- Full Angle-VQC circuit setup exactly matching authors’ simulator details.
- Multi-seed evaluation and compute setup matching the paper environment.

---

## References

- Innan et al. (2026) — benchmark protocol and reported metrics.
- Vlasic and Aguinaga (2025) — QuOp details to refine Algorithm 1 choices.
- Sato et al. (2024) — QWalkVec exact walk construction.
- Thabet et al. (2024) — QPE exact operator and anchor choices.

