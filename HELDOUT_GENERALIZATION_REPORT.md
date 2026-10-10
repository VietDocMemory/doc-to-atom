# Held-out generalization report

## Conclusion: PARTIAL

The small document-disjoint experiment passes data integrity, routing, irrelevant-signal, and
runtime checks, but it does **not** demonstrate answer-generation generalization. Doc2Atom reaches
validation/test routing Recall@8 of 1.0, yet its answer word F1 is 0.000/0.048 and is substantially
below Base + context (0.778/0.607). This is therefore `PARTIAL`, not a paper-scale result and not a
reason to start full training yet.

## Annotation and semantic-integrity audit

The cached annotation was produced by OpenAI `gpt-4.1-mini-2025-04-14`; replacing MiniMax-M2.5 is
a `REPRODUCTION_DEVIATION`. Its prompt is reconstructed rather than official. No API credential is
stored in the cache or logs.

The reported case was confirmed before correction:

- record `squad_validation_0`, question `56d9895ddc89441400fdb50e`
- question: “The name of the NFL championship game is?”; answer: “Super Bowl”
- previous `gold_atom_ids=[]`
- audited correction: `gold_atom_ids=["atom_1"]` and `relevant_atom_ids=["atom_1"]`
- grounding: `atom_1` explicitly says that Super Bowl 50 was an American football game that
  determined the NFL champion.

The migration is narrowly guarded by record/question/answer/atom preconditions. It does not infer
or rewrite uncertain mappings. Before the correction, the full 22-document cache had one semantic
error and four grounding warnings; afterward it has zero errors and four warnings. The warnings are
retained for review: two Notre Dame questions (one coreference-dependent and one 0.75 token
coverage) and two duplicate Super Bowl theme questions (0.857 token coverage). Deterministic
schema/span validation remains clean: 246 atoms, zero parser warnings, zero invalid source spans.

The new semantic validator reports, without silently repairing:

- every answerable question must have at least one gold atom;
- every irrelevant question must have no gold atoms;
- every referenced atom ID must exist;
- `relevant_atom_ids` must equal the ordered union of gold and supporting IDs;
- answer-to-gold grounding is checked where lexical grounding applies, with uncertain/coreference
  cases emitted as warnings.

## Abstraction-level audit

The 246 cached atoms comprise 240 `evidence` (97.56%), 6 `hybrid` (2.44%), and 0 `abstract`.
SQuAD's short, extractive factoid passages legitimately favor evidence-level atoms, but this is not
the whole explanation: only 80 atoms (32.5%) are exact normalized source-span matches, while mean
atom/span lexical F1 is 0.845 and 77 atoms are below 0.8. The reconstructed prompt's broad
“closely preserves” evidence definition therefore also appears to bias labels toward `evidence`.
No labels were changed to manufacture a target distribution; this remains an annotation-quality
concern for a future prompt-only study.

## Document-disjoint protocol

Seed `20261011` produced 21 eligible documents with SHA-256 context hashes: 15 train, 3 validation,
and 3 test. Pairwise context-hash intersections are empty. Each document contributes one clean
answerable question and one generated irrelevant probe, for 42 questions total and 236 atoms.
`squad_train_19` is explicitly listed as excluded because its cached record has zero QA pairs.
The selected subset validates with zero semantic errors, zero semantic warnings, and zero invalid
spans. The test split was evaluated once after all 45 optimizer steps and was never used for
optimization, early stopping, configuration, or hyperparameter selection.

Training used the final in-memory state after three epochs; the saved best-validation checkpoint
was not selected using test results.

## Held-out results

Answer metrics apply only to answerable questions. Refusal F1 applies only to generated irrelevant
probes. Each split contains 3/3 or 15/15 examples respectively, so held-out estimates have high
sampling uncertainty.

| Split | Method | Word F1 | ROUGE-L | Refusal F1 |
|---|---|---:|---:|---:|
| train/seen | Base query-only | 0.155 | 0.155 | 0.000 |
| train/seen | Base + context | **0.897** | **0.897** | 0.000 |
| train/seen | Doc2Atom | 0.171 | 0.171 | **0.379** |
| validation/unseen-doc | Base query-only | 0.000 | 0.000 | 0.000 |
| validation/unseen-doc | Base + context | **0.778** | **0.778** | 0.000 |
| validation/unseen-doc | Doc2Atom | 0.000 | 0.000 | **0.133** |
| test/unseen-doc | Base query-only | 0.194 | 0.194 | 0.000 |
| test/unseen-doc | Base + context | **0.607** | **0.607** | 0.000 |
| test/unseen-doc | Doc2Atom | 0.048 | 0.048 | **0.267** |

Base + context is the clear answer-quality control. Doc2Atom improves only slightly over query-only
on seen answers and degrades on the two unseen-document splits. Generated outputs such as `un 1999`,
`un999`, and partial refusal strings show that the short run learned a strong “unanswerable” token
prior that interferes with answer decoding.

### Routing and adapter behavior

| Split | Recall@1 | Recall@4 | Recall@8 | MRR | Answerable norm | Irrelevant norm |
|---|---:|---:|---:|---:|---:|---:|
| train/seen | 0.533 | 0.933 | 0.933 | 0.721 | 0.466 | 0.455 |
| validation/unseen-doc | 0.667 | 1.000 | 1.000 | 0.833 | 0.514 | 0.504 |
| test/unseen-doc | 0.667 | 0.667 | 1.000 | 0.708 | 0.536 | 0.571 |

Routing transfers to unseen documents: all held-out gold atoms occur in the top eight, and two of
three are top one in each held-out split. However, irrelevant adapter norms are not consistently
lower than answerable norms (notably test 0.571 versus 0.536). The refusal signal affects generated
text, but suppression is not cleanly calibrated out of distribution. The router has no explicit
null route, so irrelevant queries still receive normalized routing weights; suppression operates
through composed factor magnitude.

## Learning curves and overfit gap

Epoch means show successful optimization of the training objective:

| Epoch (steps) | Total | LM | Distill | Route | Irrelevant | Protection | Train R@1 | Train R@4 | Adapter norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 (1–15) | 32.815 | 3.730 | 4.846 | 0.360 | 238.665 | 1.629 | 0.589 | 0.889 | 1.090 |
| 2 (16–30) | 10.129 | 1.822 | 4.121 | 0.352 | 38.194 | 0.277 | 0.600 | 0.911 | 0.520 |
| 3 (31–45) | 7.103 | 1.225 | 3.331 | 0.334 | 21.838 | 0.216 | 0.533 | 0.911 | 0.458 |

The table reports unweighted component values; the total applies the configured coefficients.
Reranking stayed about 0.983 overall, sparsity about 0.99, conflict was zero, and composition and
confidence terms were small. Validation LM at steps 8/16/24/32/40 was
3.339/2.069/2.388/2.746/1.949. Validation Recall@1 remained 0.667 and Recall@4 remained 1.0.

Compared with the earlier duplicated-example overfit PASS (correct memorized prediction, R@1=1.0,
and strong irrelevant-norm reduction), the held-out run has a large generation gap. Optimization
and routing are functional; the compact training regime does not teach the compiler/composer to
reliably transfer answer content into Gemma's generations.

## Efficiency and runtime

| Method/run | Compile latency | Query latency | Peak CUDA allocation |
|---|---:|---:|---:|
| Base query-only | n/a | 1,130–1,150 ms | 5.62 GiB |
| Base + context | n/a | 905–962 ms | 5.69 GiB |
| Doc2Atom | 38.15 ms/document | 1,391–1,399 ms | 5.92 GiB |
| Training | included in training | n/a | 10.53 GiB allocated / 12.07 GiB reserved |

Training took 85.35 seconds on an NVIDIA GeForce RTX 3060. Compile latency averages 21 documents;
query latency ranges show the three split means. Base + context can be faster than query-only here
because greedy generation stops at different output lengths; these timings are diagnostic, not a
controlled throughput benchmark.

## Reproduction deviations and assumptions

- OpenAI `gpt-4.1-mini-2025-04-14` replaces MiniMax-M2.5 annotation.
- Loss coefficients and sparse density `rho=0.25` are explicit resource-profile assumptions.
- The diagnostic starts at curriculum step 2000 (K=4), disables warmup, uses 15 training documents,
  two questions per document, three epochs, and a 45-step effective run rather than paper scale.
- Gemma/compiler run on GPU; the frozen BGE reranker stays on CPU to fit RTX 3060 memory.
- FlashAttention-2 is unavailable, so attention uses SDPA.
- Gradient checkpointing is disabled because recomputation outside the dynamic-adapter activation
  context would silently omit composed memory.
- One answerable and one irrelevant question per document are a balanced diagnostic sampling
  assumption. Greedy decoding is capped at 16 new tokens.
- The implementation does not change compiler, router, or composer architecture. No full-dataset or
  paper-scale training was run.

Artifacts: `data/squad_heldout_manifest.json`, `data/squad_heldout_small.jsonl`, and
`outputs/heldout_generalization_v1/` (manifest, checkpoints, step logs, raw predictions, results).
