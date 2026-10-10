# Small real-data validation and overfit report

## Conclusion: PASS

The implementation can overfit a deliberately tiny real SQuAD slice without changing the
Doc-to-Atom compiler, router, or composer architecture. Across 20 optimizer steps, answerable
LM/distillation losses fell, the gold atom moved to rank 1, the irrelevant adapter signal was
strongly suppressed, and both diagnostic predictions became correct.

This is an implementation sanity check, not a generalization result or paper reproduction metric.

## Prompt validation on the original three documents

The reconstructed OpenAI prompt now defines `semantic_type` and `abstraction_level` as two
independent axes, gives definitions for every legal value, includes a valid XML example, and
forbids semantic labels in `abstraction_level`.

| Metric | Previous prompt | Improved prompt |
|---|---:|---:|
| Documents | 3 | 3 |
| Atoms | 28 | 42 |
| Abstraction distribution | abstract=8, evidence=20 | evidence=40, hybrid=2 |
| Parser repairs / warnings | 19 / 19 | 0 / 0 |
| Invalid source spans | 0 | 0 |

Eleven of twelve original QA mappings changed. The changes were primarily finer atom boundaries
or explicit multi-atom support: for example, the Saint Bernadette question moved from a compound
Lourdes atom to a dedicated appearance atom; the Super Bowl winner question moved from one
compound result atom to a result atom plus the two team identity atoms. All new relevant IDs
exist, every answerable QA has a gold atom, and every generated irrelevant probe has no gold atom.

Files are cached as `data/squad_small_atoms_openai.jsonl` (before) and
`data/squad_small_atoms_openai_prompt_v2.jsonl` (after).

## Cached 22-document subset

The prepared subset contains 22 documents (21 train, 1 source validation), derived from 100
SQuAD train rows and 30 validation rows. Its cached annotation contains 246 atoms and 166 QA
records (124 original questions and 42 generated irrelevant probes). Final deterministic
validation reports zero warnings and zero invalid spans. The cached file is
`data/squad_overfit_atoms_openai_prompt_v2.jsonl`; `--resume` avoids repeated API calls, while
`--retry-invalid` replaces only invalid cached records.

One document initially contained two non-verbatim ellipsis spans. Parser/schema behavior was not
changed. A single provider-side decomposition retry regenerated that record with contiguous
verbatim spans, after which the full cache validated cleanly.

## Overfit protocol

- Base configuration: `configs/resource_constrained.yaml` via `configs/small_overfit_20.yaml`.
- Model: frozen `google/gemma-2-2b-it`; 194,620,953 trainable Doc-to-Atom parameters.
- Source: real `squad_train_4`, four atoms.
- Per step: one answerable QA plus one generated irrelevant probe.
- Twenty duplicate train records and one intentional validation copy of the same examples.
- Twenty optimizer steps, AdamW, assumed resource-profile loss coefficients and `rho`.
- Paper curriculum Stage 2 (`initial_curriculum_step=2000`, K=4) so routing weights and entropy
  are observable; warmup is disabled because a 500-step warmup would make a 20-step diagnostic
  ineffective.
- Full training and the full dataset were not run.

## Validation trend by optimizer step

### Answerable question

| Step | Total | LM | Distill | Route | R@1 | R@4 | MRR | Entropy | Adapter norm | Prediction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 4.4869 | 1.5000 | 2.5312 | 0.4317 | 0 | 1 | 0.50 | 1.3862 | 2.5678 | `Learning Learning Resource Center` |
| 5 | 2.3968 | 0.6875 | 1.1406 | 0.4331 | 0 | 1 | 0.50 | 1.3832 | 0.8327 | `The Learning Resource Center` |
| 10 | 1.6360 | 0.3965 | 0.8047 | 0.4103 | 0 | 1 | 0.50 | 1.3726 | 0.5278 | `The Learning Resource Center` |
| 15 | 1.4162 | 0.3145 | 0.6836 | 0.3971 | 1 | 1 | 1.00 | 1.3675 | 0.4392 | `The Learning Resource Center` |
| 20 | 1.3759 | 0.3027 | 0.6602 | 0.3942 | 1 | 1 | 1.00 | 1.3663 | 0.4233 | `The Learning Resource Center` |

At step 20 the router ranks the gold `atom_4` first with weight 0.3029. The remaining weights are
0.2950, 0.2128, and 0.1893. R@1 and MRR therefore reach 1.0 while R@4 remains 1.0.

### Generated irrelevant probe

| Step | Total | LM | Distill | Route | Irrelevant loss | Protection | Entropy | Adapter norm | Prediction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 127.5464 | 2.3906 | 1.8906 | 0.3437 | 1229.0414 | 6.6154 | 1.3191 | 2.5720 | `answerable` |
| 5 | 22.7641 | 0.4473 | 1.7969 | 0.3520 | 201.5730 | 0.6223 | 1.3170 | 0.7889 | `unanswerable` |
| 10 | 9.3474 | 0.4785 | 1.4922 | 0.3435 | 70.2303 | 0.2092 | 1.3318 | 0.4573 | `unanswerable` |
| 15 | 6.1348 | 0.5000 | 1.2969 | 0.3377 | 39.9014 | 0.1255 | 1.3328 | 0.3543 | `unanswerable` |
| 20 | 5.5803 | 0.5195 | 1.2188 | 0.3367 | 34.9511 | 0.1114 | 1.3328 | 0.3337 | `unanswerable` |

The irrelevant suppression loss falls by about 97.2% and adapter norm by about 87.0%. The router
still has a softmax sum of 1.0 for irrelevant queries, so suppression acts through memory-factor
norm rather than a null routing mass; this is the known paper inconsistency documented elsewhere.

## Runtime and reproducibility deviations

- OpenAI `gpt-4.1-mini-2025-04-14` replaces MiniMax-M2.5 for all three annotation passes.
- Literal prompts and one invalid-span decomposition retry are reconstructed behavior.
- Loss coefficients, sparse density `rho`, metadata encoding, and several exact loss forms remain
  explicit assumptions from the resource profile.
- This overfit run used an RTX 3060 with PyTorch 2.11.0+cu128. Gemma/compiler ran on GPU; frozen
  BGE reranking ran on CPU to preserve VRAM.
- FlashAttention-2 was unavailable, so attention used SDPA. The earlier real-data smoke used the
  old CPU-only PyTorch environment; this overfit run did not.
- Gradient checkpointing was disabled because checkpoint recomputation would occur outside the
  dynamic adapter-hook context and silently omit the composed memory.
- Peak CUDA allocation was 11,126,677,504 bytes (10.36 GiB); peak reservation was
  11,398,021,120 bytes (10.62 GiB). Runtime was 41.06 seconds.
- Validation deliberately reuses the same two examples to test memorization, not held-out quality.

Detailed step logs and checkpoints are cached under `outputs/small_overfit_gpu_20/`.
