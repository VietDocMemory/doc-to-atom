# Doc-to-Atom Full-Pipeline Readiness Design

> Historical design, not a current implementation contract. Multi-provider
> annotation has been superseded by MiniMax-only HTTP transport. Consult
> `../../MINIMAX_MIGRATION_PAPER_AUDIT.md` for verified capabilities and gaps.

Date: 2026-10-17  
Status: approved for planning; the user delegated remaining design decisions  
Production target: Molab Linux multi-GPU  
Local target: Windows RTX 4070 Ti SUPER smoke and integration tests only

## Purpose

Turn the current research prototype into a stable, reproducible, end-to-end pipeline that can later run full training and full evaluation without claiming that unpublished reconstruction choices are paper facts. The work prioritizes correctness, explicit provenance, deterministic resume, validation-only model selection, complete evaluation traces, and recoverable artifacts. This phase must not start a full training or full atomization run.

## Non-goals

- Reproduce unpublished paper details by guessing silently.
- Tune on or repeatedly inspect the test split.
- Enable Micro-KV without a declared experimental specification.
- Change the architecture merely to improve headline metrics.
- Run the full six-dataset atomization, full training, or full test suite on Molab during implementation.
- Replace the custom teacher/student workflow with Hugging Face Trainer or a new training framework.

## Current-state audit

### What already works

- The repository packages cleanly with `uv`; 23 existing tests pass.
- The local Windows environment uses Python 3.12 and a CUDA-enabled PyTorch build.
- Core components exist for atom schemas, XML parsing, semantic checks, a shared early-exit encoder, compiler heads, two-stage routing, composition, dynamic LoRA injection, curriculum losses, a small custom training loop, memory serialization, and query inference.
- The checked-in SQuAD smoke data contains 3 records, 28 atoms, and 18 questions. Structural validation reports no invalid source spans, and the separate semantic audit reports no current errors or warnings.
- Paper-unknown values are generally visible in `PAPER_SPEC.md`, `REPRODUCTION_NOTES.md`, and the paper config.

### Material gaps and inconsistencies

1. `training.cli` has no `--resume` or `--load-weights` path.
2. Checkpoints store only module weights, optimizer state, and one ambiguous `step`; they omit scheduler, scaler, epoch, sampler cursor, early-stopping state, curriculum state, RNG, config/data hashes, and W&B identity.
3. The current `global_step` increments per document record, while optimizer and scheduler advance only after gradient accumulation. Curriculum, validation cadence, and `max_steps` therefore do not share optimizer-step semantics.
4. The scheduler's total-step calculation ignores world size and accumulation.
5. There is no deterministic shuffling/sampler state, resumable dataloader, distributed training integration, or complete Molab launch path.
6. Gradient checkpointing is requested by `resource_constrained.yaml` but silently disabled with a warning because the hook-scoped adapter is absent during recomputation.
7. Early stopping exists only as an epoch-end comparison of validation LM. It has no `min_delta`, no persisted patience state, and no robust NaN policy.
8. The evaluation CLI only rescored an existing prediction JSONL. It does not load a checkpoint, compile/load memory, route, compose, generate, or produce complete metrics and efficiency statistics.
9. End-to-end evaluation logic exists only in specialized experiment scripts and imports private training helpers, duplicating behavior instead of providing a supported evaluator.
10. F1/ROUGE currently use only the first gold answer. Refusal is measured as word-F1 against the literal string `unanswerable`, not as a dataset-level classification metric.
11. Routing softmax always sums to one. Irrelevant suppression can reduce adapter factors but cannot make routing mass near zero as the paper prose suggests.
12. Micro-KV tensor heads and composition exist, but injection semantics do not. `factory.py` correctly refuses to enable it, while documentation overstates component completeness.
13. Memory files lack checkpoint/config/atomization provenance and can be reused with incompatible weights.
14. Dataset preparation supports only a small SQuAD subset even though preprocessor stubs exist for the other benchmarks.
15. Dataset versions, split policies, source hashes, preprocessing versions, exclusions, and document-disjointness are not captured by a common manifest.
16. Atomization couples an OpenAI-compatible SDK directly to the workflow. Retry is partial and provider-specific; cache is only final-output `record_id` reuse, not stage-level content-addressed caching.
17. Context and generated fields are interpolated into XML-like prompts without robust escaping. Recovery parsing can salvage elements but lacks structured failure codes and raw-response provenance.
18. Structural validation mutates annotations by dropping references and clearing fields, making it difficult to distinguish provider output from deterministic repair.
19. Prompt provenance is a string rather than a versioned prompt bundle with semantic version and hash.
20. There is no W&B dependency or unified tracking lifecycle.
21. There is no hyperparameter tuning framework or validation-only selection manifest.
22. Required regression tests for full resume equivalence, gradient flow across the full path, forced-gold evaluation, one-example overfit, and small held-out end-to-end evaluation are missing or incomplete.
23. `docs/PHASE_LOG.md` and `docs/IMPLEMENTATION_MAP.md` describe training/evaluation/resume as complete even though the supported CLIs and artifacts do not meet that claim.
24. Runtime dependencies drift from the documented paper environment: the project requires `torch>=2.6`, while the current Linux lock resolves a much newer PyTorch than the PyTorch 2.1 paper runtime. Molab dependencies need an explicit tested profile rather than an unbounded lower constraint.
25. Previously reported real-data artifacts under `outputs/real_data_smoke/` are absent from the current workspace because outputs are ignored. Reports must distinguish retained source-controlled evidence from ephemeral local artifacts.

## Architectural approach

Keep the custom teacher/student training design, but separate orchestration from the mathematical components. Use Accelerate for distributed coordination and DeepSpeed ZeRO-2 for the Molab production profile. Do not migrate the algorithm to Hugging Face Trainer, Lightning, Hydra, or Ray.

The supported stage graph is:

```text
prepare -> atomize -> validate -> train -> evaluate
                                      \-> tune
```

Each stage consumes an artifact plus its manifest and emits a new versioned artifact plus a manifest. File names alone are never sufficient provenance.

New responsibility boundaries:

- `doc_to_atom.config`: typed schema, inheritance, profile classification, validation, canonical serialization, fingerprints, and resume compatibility.
- `doc_to_atom.runtime`: run context, deterministic seeding, environment capture, structured event logging, W&B lifecycle, signals, and rank-aware operations.
- `doc_to_atom.artifacts`: atomic JSON/JSONL/safetensors I/O, hashes, schemas, committed-directory protocol, and artifact indexes.
- `doc_to_atom.data`: dataset registry, adapters, split policies, normalized raw schema, manifests, streaming, and mixture construction.
- `doc_to_atom.atomization`: provider protocol, prompt registry, staged cache, retry policy, parsing, canonicalization, validation, quarantine, and consolidation.
- `doc_to_atom.training`: serializable trainer state, checkpoint manager, deterministic sampler, distributed runner, validation loop, and early stopping.
- `doc_to_atom.evaluation`: supported end-to-end evaluator, baselines, routing/generation metrics, efficiency/statistics, resumable prediction shards, aggregation, and reports.
- `doc_to_atom.tuning`: Optuna study definition, typed search spaces, subprocess trials, pruning, selection manifest, and W&B grouping.

Thin scripts remain for backward compatibility and call package CLIs.

## Configuration model

Configurations are separated into four namespaces:

1. `configs/paper_like/`: only disclosed paper values. Unknown values remain null and make training fail validation.
2. `configs/reconstruction/`: runnable choices for every unknown, each linked to an assumption ID and rationale.
3. `configs/experiments/`: overfit, ablation, tuned, null-route, and resource-constrained variants. They may not be labeled paper reproductions.
4. `configs/runtime/`: local Windows smoke and Molab Accelerate/DeepSpeed execution profiles. Runtime profiles may alter placement, precision, workers, accumulation, and kernels, but not algorithm semantics silently.

The resolved configuration is validated before model allocation. A canonical copy and SHA-256 fingerprint are saved. Resume rejects changes to architecture, optimizer, schedule, data, curriculum, prompts, or seed. A small explicit allowlist permits operational changes such as log frequency and W&B connectivity mode.

Every assumption has a stable ID, category, description, chosen value, rationale, affected metrics, and source status. Every experimental feature has a deviation ID. Reports list active IDs.

## Run and artifact contract

Every `prepare`, `atomize`, `train`, `evaluate`, and `tune` invocation creates a run directory:

```text
runs/<run-id>/
  config/{requested.yaml,resolved.yaml,assumptions.json}
  manifests/{run.json,environment.json,dataset.json,atomization.json}
  logs/{events.jsonl,warnings.jsonl}
  checkpoints/{step-*/,last.json,best.json}
  memory/{index.json,*.safetensors}
  predictions/<split>/rank-*.jsonl
  metrics/*.json
  reports/*.md
  wandb/
```

Important artifacts are written to a temporary sibling and atomically renamed. Distributed checkpoint directories receive a `COMMITTED` marker only after all ranks succeed. Pointer files refer only to committed checkpoints. Rank 0 owns scalar logs, reports, merged predictions, and W&B; all ranks participate in distributed state and RNG checkpointing.

The run manifest records command, timestamps, host, Git commit when available, package versions, Python, CUDA, GPU topology, world size, precision, input/output hashes, config fingerprint, seeds, active assumptions/deviations, and W&B identity. It never records secrets.

## Training semantics

Counters have distinct meanings:

- `micro_step`: completed document batches that ran forward/backward.
- `optimizer_step`: completed optimizer updates.
- `curriculum_step`: normally `optimizer_step`; diagnostic configs may declare a visible offset.

Scheduler, warmup, curriculum, validation cadence, checkpoint cadence, early stopping, and `max_steps` use optimizer steps. Effective batch is validated as microbatch times accumulation times world size.

The training unit is a document plus a bounded list of its QA examples. QA loss is averaged per document; document loss participates in accumulation. A deterministic distributed sampler derives each epoch permutation from the global seed and epoch. Checkpoints store epoch and sampler cursor so resume recreates the order and skips exactly the completed units.

Checkpoints are committed only at optimizer-step boundaries. A termination signal requests a checkpoint after the active accumulation window finishes. This avoids pretending resume equivalence when partial gradient buffers were not persisted.

### Checkpoint v2 contents

- Trainable encoder projection, compiler, and router state.
- Optimizer, scheduler, and mixed-precision scaler state.
- Epoch, sampler cursor, micro step, optimizer step, and curriculum step/stage.
- Best validation metric, patience counter, and early-stopping status.
- Python, NumPy, Torch CPU, and per-rank CUDA RNG state.
- Resolved-config, dataset-manifest, model-revision, and prompt fingerprints.
- World size, distributed backend metadata, W&B run ID and W&B mode.

Accelerate/DeepSpeed state supports exact Molab resume. A portable safetensors weights artifact supports evaluation. Legacy v1 checkpoints support weights-only migration but never claim exact optimizer/RNG resume.

Early stopping declares metric, direction, patience, and minimum delta. It observes validation only. NaN/Inf never counts as improvement and produces a structured warning. Both `last` and `best` are retained; evaluation defaults to `best`.

### Gradient checkpointing

The current hook activation is unsafe during checkpoint recomputation because its context has ended before backward. Until adapted and forced-gold recomputation equivalence is demonstrated, supported profiles set gradient checkpointing false and fail fast if it is requested. Molab initially relies on 8 x A100 80 GB, bf16, FlashAttention-2, ZeRO-2, and the documented microbatch/accumulation settings. A future implementation may enable checkpointing only after tests show matching logits and gradients for every adapter target and forced-gold path.

## Routing, refusal, and Micro-KV policy

The paper-like router retains top-K softmax weights that sum to one. Reports explicitly state that irrelevant suppression can reduce the composed adapter norm but cannot make total routing mass near zero.

A separate reconstruction/experimental null route may add a trainable null-memory logit or relevance gate. Atom weights remain conditional softmax weights; total adapter mass is multiplied by a relevance probability. The null route is disabled in paper-like configs, carries a deviation ID, and calibrates any refusal threshold on validation only. Paper-like and null-route results are always reported separately.

Micro-KV stays disabled. The project may serialize and compose its provisional tensor container for compatibility tests, but no training or evaluation profile may inject Micro-KV until slot count, head mapping, positional behavior, injection site, and initialization are specified in an explicitly experimental design. Enabling an incomplete Micro-KV config is a hard error.

## End-to-end evaluation

Validation and test are separate commands. Test evaluation requires a finalized selection manifest containing the immutable checkpoint, config, dataset, prompt, and calibration hashes. Training never loads test records.

For each document, the evaluator loads a compatible checkpoint, loads or compiles a provenance-checked memory bank, processes each query through encoder, stage-1 retrieval, frozen reranker, metadata bias, top-K selection, composition, dynamic LoRA generation, and metrics. Memory cache keys include checkpoint, record/context, atomization, and compiler fingerprints.

Supported methods are base query-only, base full-context, Doc2Atom predicted routing, and Doc2Atom forced-gold diagnostic. Forced-gold is labeled an upper-bound diagnostic and never mixed with primary results.

Answer metrics use maximum score across all gold references. Reports include word F1, ROUGE-L, refusal precision/recall/F1, answerable false-refusal rate, irrelevant false-answer rate, strict-gold Recall@1/@4/@8/@K, relevant-union Recall@K, MRR, missing-gold counts, and forced-gold coverage. Irrelevant rows do not enter retrieval denominators.

Timing separates model/checkpoint load, memory compile, memory cache load, query encoding, stage-1 retrieval, reranking, composition, generation, and total time. CUDA measurements synchronize boundaries and record allocated and reserved peaks. Aggregates include mean, median, p90, and p95 with an explicit warm-up policy.

Adapter statistics include A/B norms, low-rank or chunked Frobenius norm of scaled gated delta-W, gate min/mean/max, fraction below declared thresholds, per-layer/module summaries, routing entropy, maximum routing weight, effective atom count, and relevance/null mass where applicable. Every tensor statistic has non-finite checks.

Evaluation writes stable per-example keys and rank shards. Resume skips committed keys. Merge detects duplicate and missing examples and orders outputs deterministically.

## Atomization design

An `AtomizationProvider` protocol returns normalized responses containing provider, requested/resolved model, provider request ID, content, token usage, latency, finish reason, and attempt. MiniMax and OpenAI are independent adapters; core workflow code does not import a provider SDK.

The three stages—decomposition, irrelevant-probe generation, and question annotation—have independent content-addressed caches. Cache keys include provider, resolved model, prompt version/hash, decoding parameters, input hash, and dataset-adapter version. A failed question stage can resume without repeating a valid decomposition stage.

Retries classify rate limit, transient transport/server error, invalid request, validation failure, and terminal error. Retryable errors use bounded exponential backoff with jitter. Every attempt is recorded. A failure ledger supports selective retry.

Each record and stage is written atomically as raw response, parsed output, canonicalized output, and validation report. Final JSONL is a deterministic consolidation product, not the primary mutable cache.

Prompts have semantic versions, SHA-256 hashes, provenance status, dataset-adapter versions, and changelogs. XML-bound inputs are escaped or wrapped safely. Parsing uses a safe XML implementation with size/depth bounds. Recovery never overwrites raw output and always emits structured issue codes.

Validation is separated into parsing, canonicalization, structural validation, source-span validation, and semantic validation. Raw provider fields are preserved. Every deterministic repair has a transformation code. Source spans record contiguous character offsets; whitespace-normalized matching is a declared reconstruction deviation, and ambiguous matches are warnings. Invalid records are quarantined or fail according to an explicit policy. Atom-only records are ineligible for QA training until annotation completes.

Before a full atomization command, the CLI requires a dataset manifest, cost/call estimate, canary result, configured validation thresholds, and explicit `--confirm-full-run`.

## Dataset workflow

The registry pins these current Hub repositories and resolves a concrete commit at preparation time:

- `rajpurkar/squad`
- `ucinlp/drop`
- `allenai/ropes`
- `framolfese/2WikiMultihopQA`
- `allenai/qasper`
- `zai-org/LongBench`

Adapters normalize each source into records with dataset/revision/subset/split, stable record ID, source document ID, context, questions, answers, and source metadata. IDs prefer upstream IDs and otherwise use content hashes; row indices alone are forbidden.

The dataset manifest captures repository commit, subset, original splits, split policy, preprocessor version, seed, row/document/question counts, exclusions, duplicate handling, document hashes, output hashes, and licensing metadata. If labeled test data is unavailable, an explicit document-disjoint split policy derives validation/test from a labeled source split and records that reconstruction assumption. Validation and test documents may not overlap.

SQuAD, DROP, ROPES, 2WikiMultiHopQA, and QASPER feed the configurable training mixture. LongBench remains evaluation-only and enumerates enabled subsets in config. Preparation supports full, streaming, and bounded smoke modes.

## W&B lifecycle

Every prepare, atomize, train, evaluate, and tune run creates a local run context and a W&B run. If authentication or connectivity is unavailable, it automatically uses offline mode under the run directory. Resume reuses the same W&B run ID. Local artifact failure is fatal; W&B transport failure is not.

Config, manifests, scalar logs, checkpoints, predictions, metrics, and reports are logged. Large artifacts follow configurable checkpoint/end-of-run policies to avoid redundant uploads. Dataset and annotation artifacts may contain document/question text, so the guide requires a private project for confidential data. Secrets are redacted before logging. Offline runs include exact `wandb sync` instructions.

## Hyperparameter tuning

Optuna with a persistent SQLite journal is the default local/Molab study engine. Each trial launches the normal training command in an isolated subprocess and run directory, rather than importing and mutating a global trainer. Trials form one W&B group and retain their own manifests, checkpoints, metrics, and failure reason.

The typed search space supports loss weights, learning rate, LoRA rank/alpha, memory layers, routing K/temperature, sparse density, and refusal calibration. Structural parameters that change checkpoint shape are validated before launch. Invalid combinations are rejected with reasons.

Selection uses a configured validation objective or explicit multi-objective tuple. Pruning begins only after a configured minimum number of validation observations. Test evaluation is unavailable inside a trial. Study completion emits a ranked trial table, best resolved config, Pareto data when applicable, and a selection manifest that locks the chosen trial before one final test evaluation.

## Test strategy

Pytest markers separate `unit`, `integration`, `gpu`, `network`, `slow`, and `molab`. The default local suite uses no API calls and no gated model download.

Required coverage:

- Exact LoRA injection math, scoping, cleanup, layer/module targeting, dtype/device behavior, and no base-weight mutation.
- Gradient flow from LM/routing/composition objectives through gates, compiler heads, router, and encoder projection while base and reranker remain frozen.
- Continuous-versus-resumed training equivalence, including optimizer, scheduler, sampler order, curriculum, early stopping, RNG, and W&B identity.
- Routing top-K, temperature, metadata bias, softmax baseline, irrelevant behavior, null route when enabled, and no gold leakage.
- Forced-gold composition and generation, clearly separated from predicted routing.
- One-example overfit acceptance test with a tiny local model and bounded steps.
- Small document-disjoint held-out evaluation producing all required artifacts and metrics.
- Atomization provider contracts, retry/backoff, stage-cache invalidation, resume, XML escaping/recovery, source offsets, semantic quarantine, prompt hashes, and dataset manifests.
- Evaluation resume/merge, multi-reference metrics, refusal confusion matrix, timing schema, memory provenance, and adapter statistics.

GPU/Molab smoke tests verify bf16, FlashAttention fallback policy, DeepSpeed launch, distributed checkpoint round-trip, and portable evaluation weights without running full training.

## Documentation deliverables

- `IMPLEMENTATION_AUDIT.md`: evidence-backed gap list, resolution status, assumptions, and remaining blockers.
- `PRE_FULL_RUN_CHECKLIST.md`: credentials, dataset/prompt revisions, canary thresholds, storage, Molab topology, resume drill, W&B offline sync, validation lock, and no-test-leak checks.
- `FULL_TRAINING_GUIDE.md`: environment setup and exact prepare, atomize, train, resume, validation, and final test commands.
- `TUNING_GUIDE.md`: search-space design, study launch/resume, pruning, selection, W&B grouping, and finalization commands.

Existing phase documentation is corrected so “implemented interface,” “tested smoke path,” and “full-run ready” are distinct statuses.

## Full-run readiness gate

The repository is ready for a future full run only when all of the following are true:

1. Fast tests and required integration tests pass.
2. Resume equivalence passes from a fresh process.
3. A Molab multi-GPU smoke completes and resumes from a committed checkpoint.
4. Dataset manifests resolve pinned revisions and prove split policy/integrity.
5. Atomization canary meets configured structural, span, semantic, and failure-rate thresholds.
6. The evaluation smoke emits every required metric, statistic, prediction, and report artifact.
7. W&B online and offline-resume/sync paths are tested without leaking secrets.
8. The best-checkpoint selection manifest is created from validation only.
9. Paper-like, reconstruction, and experimental results cannot be confused by naming or metadata.
10. `PRE_FULL_RUN_CHECKLIST.md` is completed by the operator.

Passing this gate authorizes preparation for a full run; it does not itself launch one.

## Implementation sequence

1. Correct the audit/status documentation and introduce typed config, assumptions, manifests, atomic artifacts, and run context.
2. Add W&B offline-first tracking and dependency/runtime profiles.
3. Implement trainer state, checkpoint v2, deterministic sampler, resume, early stopping, and corrected optimizer-step semantics.
4. Add Accelerate/DeepSpeed Molab orchestration and distributed smoke coverage.
5. Replace the scoring-only evaluation CLI with the supported end-to-end evaluator and reports.
6. Refactor atomization into provider/stage-cache/prompt/validation layers.
7. Implement the six-dataset registry, adapters, manifests, and smoke fixtures.
8. Add Optuna tuning and selection manifests.
9. Complete tests, migration checks, guides, exact commands, and readiness checklist.

Each phase must preserve a green fast test suite. No full dataset, full atomization, full training, or final test evaluation is run during implementation.
