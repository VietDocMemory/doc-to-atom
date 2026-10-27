# Evaluation summary — fill after measured Molab runs

- Model/checkpoint path, optimizer step, base model: `...`
- Config snapshot/fingerprint, code commit, assumptions/deviations: `...`
- Source/atomized file SHA-256, split, question counts, annotation provider: `...`
- Actual hardware and PyTorch/CUDA/attention implementation: `...`
- Scoring/generation protocol and refusal target: `...`
- Comparison status (paper-comparable / partial / not comparable), reason: `...`

## Table 1 — answerable QA (F1 and ROUGE-L, %)

Paper columns are **Gemma-2-2B-It Doc2Atom** in paper Table 1; use a separate
table if evaluating Qwen. Enter measurements from the corresponding dataset's
`metrics/evaluation_test.json`. `—` means not measured, not zero.

| Dataset | Paper F1 | Paper ROUGE-L | Molab F1 | Molab ROUGE-L | N answerable |
|---|---:|---:|---:|---:|---:|
| 2WikiMultiHopQA | 39.62 | 39.62 | — | — | — |
| DROP | 28.31 | 28.31 | — | — | — |
| QASPER | 30.78 | 30.44 | — | — | — |
| ROPES | 53.52 | 53.52 | — | — | — |
| SQuAD | 59.71 | 59.71 | — | — | — |
| LongBench-2WikiMQA | 29.63 | 29.63 | — | — | — |
| LongBench-HotpotQA | 27.96 | 27.82 | — | — | — |
| LongBench-MFQA-en | 26.00 | 25.83 | — | — | — |
| LongBench-MFQA-zh | 8.57 | 8.57 | — | — | — |
| LongBench-MuSiQue | 20.86 | 20.86 | — | — | — |
| LongBench-NarrativeQA | 20.58 | 20.27 | — | — | — |
| LongBench-QASPER | 29.85 | 29.26 | — | — | — |
| LongBench-TriviaQA | 79.54 | 79.54 | — | — | — |
| Overall (only if aggregation protocols match) | 37.99 | 37.88 | — | — | — |

## Table 2 — refusal F1 on irrelevant queries (%)

| Dataset | Paper Gemma Doc2Atom | Molab refusal F1 | N irrelevant | N answerable | Comparable? |
|---|---:|---:|---:|---:|---|
| 2WikiMultiHopQA | 90.01 | — | — | — | — |
| DROP | 92.03 | — | — | — | — |
| QASPER | 58.05 | — | — | — | — |
| ROPES | 94.66 | — | — | — | — |
| SQuAD | 85.25 | — | — | — | — |
| LongBench-2WikiMQA | 97.50 | — | — | — | — |
| LongBench-HotpotQA | 90.17 | — | — | — | — |
| LongBench-MFQA-en | 95.22 | — | — | — | — |
| LongBench-MFQA-zh | 95.95 | — | — | — | — |
| LongBench-MuSiQue | 95.33 | — | — | — | — |
| LongBench-NarrativeQA | 84.33 | — | — | — | — |
| LongBench-QASPER | 61.50 | — | — | — | — |
| LongBench-TriviaQA | 64.00 | — | — | — | — |

## Table 3 — efficiency: measurement scopes must match

| Metric | Paper Gemma Doc2Atom | Molab result | Comparable? / method |
|---|---:|---:|---|
| Mean **compile/update** latency, seconds | 2.10 | — | — |
| Mean **additional compile** GPU memory, GB | 13.43 | — | — |
| Full-training peak allocated VRAM, GiB | Not reported | — | Diagnostic only |
| End-to-end training wall time, hours | Not reported | — | Diagnostic only |

Provide per-stage LM/distill/route/irrelevant/protect/sparse losses, mean
no-memory weight and adapter norm by answerable/irrelevant group, nonfinite
counts and OOM incidents. Document deviations: one Blackwell GPU instead of
8×A100; actual global batch 16 instead of 128; independently reconstructed
prompts and unpublished coefficients; data/probe and scoring differences.
