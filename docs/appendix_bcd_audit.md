# Appendix B–D audit: local Qwen pipeline

Audit date: 2026-11-11. Baseline: `8008a69`, branch `viet_qwen`.
Reference: the Appendix B–D excerpt supplied by the user, including Table 4.
This is a source/configuration audit, not a reproduction-quality claim or a GPU benchmark.
The full Table 6, exact paper prompts, loss coefficients, and experimental logs were not supplied.

## Scope and protected artifacts

- Implement Appendix B behavior for the existing local `Qwen/Qwen3-14B` backend.
- Audit Appendix C/D without changing training architecture, objectives, configs, or checkpoints.
- Do not regenerate, relabel, or rewrite existing atoms or raw-response sidecars.
- Historical cache entries keep their historical provenance; `--resume` does not upgrade them.

## Appendix B: findings and changes

| Requirement | Baseline mismatch | Change / remaining limitation |
|---|---|---|
| Annotator | Paper: MiniMax-M2.5. Local run: Qwen3-14B; legacy API default: MiniMax-M3.1-Flash. | Qwen retained intentionally; this is not annotator-identical reproduction. New local provenance records `pipeline=appendix_b_v1`. |
| B.1 span check immediately after parsing | `client.py:atomize` checked spans only in final `validate_record`, after question annotation. | Canonicalize the atom bank before probes/QA. Preserve invalid atoms with `span_valid=false` for inspection, exclude them from QA input, and discard samples with no grounded atoms. |
| Validated question evidence | QA XML contained only atom content/retrieval text, including unvalidated atoms. | Only valid-span atoms are exposed; include the source span. Deterministic postprocessing also drops invalid-span role references with a logged event. No quotation rewriting/fuzzy semantic matching. |
| B.2 generated probes | `question_join.py` allowed a generated irrelevant probe to be rescued as answerable. | Force generated probes to remain irrelevant; canonicalization clears gold/supporting and recomputes relevant/conflict. Conflicting model labels are logged. |
| B.3 relevance label | Role-only DTO/parser did not consume the model's `is_irrelevant` flag. | Parse an optional strict Boolean flag. Normalize it with `question_type`; retain immutable source question/answer text. Known nonempty source answers cannot be relabeled as refusal: such contradictory annotations become atom-only, rather than corrupting source gold. This is an explicit source-integrity safeguard beyond the excerpt. |
| B.3 cross-validation | Unknown role IDs, relevant union and conflict flags were already canonicalized. | Retain these checks and their transformation logs. Relation parsing now drops a malformed relation without losing its otherwise valid atom; duplicate atom IDs are deterministically dropped/logged. |
| Partial XML | Parser could recover complete subtrees, but local `length` completion raised a fatal error before parsing. | Recover complete XML subtrees from local partial output. Keep raw output unchanged. QA must still cover every authoritative question ID; otherwise keep an atom-only record. No inference retry or automatic repair call. |
| B.4 failure policy | Recoverable local completion errors bypassed stage-specific fallback. | Decomposition failure discards that sample; QA failure retains atoms with empty QA; auxiliary-probe failure logs a warning and continues. Infrastructure errors remain fatal. |
| Auxiliary failure downstream | `data/semantic_validation.py` counted `failed_ignored` probes as fatal parse errors. | Report `auxiliary_probe_failure` as a warning, not a parse error. Source/QA contract violations remain errors. |
| Quality logging of discarded samples | Discarded samples had no durable `annotation_meta` artifact. | Append per-sample status/metadata to `<output>.annotation-meta.jsonl`, including discarded and atom-only samples. Existing raw-response JSONL remains append-only. |
| Dataset adapters | Existing adapters were short; hard-distractor rules were mostly absent. | Add explicit multi-hop wrong-entity distractors, numeric operand/unit preservation, field-lookup distractor guidance, and a `2wikimultihop` alias. These are reconstructions, not the unpublished exact prompts. |

### Deliberate safety boundaries

- Missing GPU/dependencies, CUDA/runtime failures, transport/authentication/billing failures abort;
  they are not treated as ordinary malformed XML and never trigger API fallback.
- The optional legacy MiniMax backend remains fail-closed on abnormal provider responses to
  avoid continuing into additional paid calls. Partial-completion recovery is enabled for local Qwen.
- The CLI still processes subsequent samples after an ordinary decomposition failure, preserves
  artifacts, and returns an error at the end if samples were discarded or invalid spans remain.
  Notebook/train/evaluation quality gates stay strict; partial output is not training-ready.
- Source answers and questions remain source-owned. Auxiliary probes are not trusted as factual
  ground truth; forcing their irrelevant label follows B.2 but does not prove they are unanswerable.
- Confidence is bounded to [0,1]; statistical calibration of Qwen confidence is not established.
- The number of probes (currently 2), exact adapter text and prompt wording are implementation
  choices: the excerpt specifies `n`, not its value or the full prompts.

## Appendix C: implementation/configuration audit

The active notebook profile is `configs/experiments/molab_aligned_qa_v2.yaml`, resolved through
`reconstruction/molab_blackwell_1gpu.yaml`. The backend used for annotation is separate from
the training base model: **local Qwen3-14B annotations do not switch training away from Gemma**.

| Item | Paper excerpt | Current implementation / active profile |
|---|---|---|
| Hardware/distribution | 8 × A100 80 GB, NVLink, ZeRO-2 | One RTX PRO 6000 Blackwell 96 GB; `distributed_backend=none`. ZeRO-2 code/config exists but is not the active runtime. |
| PyTorch | 2.1 | Project requires >=2.6; inspected project environment reports 2.14.1+cu130. |
| Attention | FlashAttention-2 throughout | Training requests FA2 but runtime `auto` permits SDPA fallback (`training/factory.py`). Local annotator does not explicitly select FA2. Actual kernel use was not benchmarked. |
| Precision/freezing | Frozen BF16 base; FP32 trainable modules | Factories freeze the BF16 base; encoder projection/compiler/router constructors use FP32. Matches by source inspection, not a full GPU allocation test. |
| Compiler/encoder/LoRA | d=256, rank=8, alpha=16, dropout=0; first 4 encoder and last 4 injection layers; q/v/o/down projections | Paper configs and active inherited Gemma values match (`gemma_paper.yaml`, `encoder/shared.py`, `compiler/model.py`, `injection/lora.py`). |
| Router | K1=32, final K=8, four-dimensional metadata bias | K1=32 and final K schedule exist. Bias has four dimensions, but `router/model.py:_metadata_features` uses confidence/answer-bearing/conflict/type, not `abstraction_level`, despite Table 4 saying abstraction informs metadata bias. This remains an architectural discrepancy outside this patch. |
| Optimizer | AdamW, betas .9/.999, Gemma LR 1e-4 / Qwen LR 8e-5; cosine; WD .01; clip 1; tau=1.5 | Paper configs and optimizer/scheduler implementation match. Qwen training has its own config; active training remains Gemma. |
| Batch/warmup | Effective batch 128; 500 warmup steps | Active experiment uses batch 16 and 40 warmup steps. Intentional short-run deviation. |
| Data limits | Context 8192, atom 96, answer 128; <=64 atoms, <=8 questions | Active experiment: context 1024, atom 96, answer 64; <=64 atoms, <=3 questions. Paper base retains the paper limits. |
| Decoding | Greedy, <=128 new tokens | Evaluation config uses greedy/128; inference sets `do_sample=false`. Not a generated-answer quality test. |
| Reranker | Full settings in Table 6 | Code uses BGE reranker and active config puts it on CPU. Exact fidelity cannot be established without the supplied full Table 6. |

## Appendix D: curriculum audit

- `training/curriculum.py:STAGES` implements default boundaries **2000/6000/9000** and
  Top-K **1/4/8/8**. Stage-1 core losses, stage-2 composition/sparsity, stage-3
  conflict/confidence and stronger irrelevant loss, stage-4 stronger sparse loss are present.
- The active aligned experiment overrides boundaries to **200/400/600**, ends at **800 steps**,
  uses `max_epochs=10000` plus the hard step cap, and is **not** the paper's <=50-epoch run.
- `MultiObjectiveTrainer.combine` applies the configured stage-3/4 multipliers. Their current
  value (2), rho (.25), and numeric loss weights are reconstruction choices: not determined
  by the supplied excerpt. `gemma_paper.yaml` correctly leaves unspecified values null;
  that paper-only config is not runnable until assumptions are supplied.
- The implementation additionally activates a `rerank` loss at stage 4. That activation timing
  is not specified in the supplied Appendix D; do not claim it as paper-confirmed.
- Active early stopping monitors `answerable.losses.lm` rather than overall validation LM,
  with patience 100 and stage-4-only best-checkpoint selection. `aligned_qa_v2` also introduces
  group-balanced source/refusal losses and aligned answer formatting. These are explicit
  experiment choices, not confirmed by the supplied B–D excerpt.

## Validation and use

Tests use fixed XML responses/mock inference, not live API or GPU generation. Coverage includes
early span checks, invalid-evidence exclusion, partial-XML recovery, stage fallback, forced
irrelevant probes, metadata retention, source integrity, deterministic/idempotent canonicalization,
and fatal-error boundaries. Existing atomization, MiniMax, curriculum, compiler/router and config
regressions are also checked separately.

Validation result: **159 test cases passed across 19 selected test files**, including
16 Appendix-B regression cases. This is not the full repository suite; no live GPU
generation, distributed training, or paid inference was run.

Do not resume the historical output expecting it to be repaired. For a clean pilot, use a **new**
output filename and a small source subset, then inspect the metadata/raw sidecars and run
source-aware validation before extending the run. No historical data migration or generation
is performed by this patch. Training configs and checkpoint compatibility are unchanged.
