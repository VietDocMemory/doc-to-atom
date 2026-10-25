# Local inference/evaluation protocol and artifacts — design

## Intent and constraints

Improve answer formatting and measurement without changing Doc-to-Atom architecture, training losses, optimizer, curriculum, source gold answers, or checkpoint fingerprint verification. Work and tests run locally; do not train the full model, call paid APIs, run Molab, push, or upload artifacts during implementation. The supplied `predictions.jsonl` is untrusted data, not instructions.

The supplied file has 10 validation predictions: 8 answerable and 2 irrelevant. The current score-only CLI reports F1/ROUGE-L 0.207005, refusal_f1 0, Recall@K 1, and MRR 0.6875. Local source and atomized JSONL match all 10 IDs/questions/gold-answer lists, but local best checkpoint is step 31 whereas supplied predictions are step 28. The local machine has 16 GB RAM, no detected CUDA GPU, and no cached Gemma base model. Thus old predictions can be re-scored, and a synthetic local train→checkpoint→evaluate smoke can run, but a real step-28 before/after generation cannot be claimed without the matching checkpoint/base model.

## Chosen approaches

- Inference: retain training-compatible `Question: ...\nAnswer:` framing, add concise-answer instruction and a shared one-line stop policy. Record policy/version and generated answer text. Do not extract a substring based on gold answers. Prompt-only was rejected because the existing model continues with explanations; arbitrary gold-aware trimming is prohibited.
- W&B: upload best **model weights** and final last **full exact-resume checkpoint** as separately aliased artifacts. Never upload every intermediate full checkpoint. Both uploads remain policy-controlled; no actual upload in this local task. Uploading both best and last full was rejected because it duplicates optimizer/RNG state and raises run transfer by roughly 1.6 GB.

## Inference contract

- A single QA formatting/stopping helper is used by Doc-to-Atom and compatible baselines. It requests one short answer or the configured refusal literal and stops after the first nonempty answer line; a leading newline alone does not terminate. Handle a token containing newline plus following text deterministically. Apply this at generation, independent of gold text.
- Keep `max_new_tokens` as a safety cap and `do_sample=False`. Continue to use model EOS. Save generation settings in output metadata. Preserve the raw model continuation separately when useful for debugging; never replace primary scoring with an analytic first-line transform of a previously generated file.
- The primary F1 scores the prediction emitted by its declared inference protocol. Legacy predictions retain their full original text. `first_line_f1` is a separate diagnostic only. A changed inference protocol is reported as such, not described as a model-quality-only gain.

## Evaluation contract

- For answerable SQuAD questions, use official SQuAD normalization (lowercase, ASCII punctuation removal, English article removal, whitespace collapse), token F1 and exact match, max over all nonempty source references. Primary values remain fractions in JSON; label percent when displayed. ROUGE-L F-measure remains a separate metric (SQuAD does not officially define ROUGE-L), likewise max over source references. Never alter source answers.
- Refusal is a binary decision under an explicit, versioned policy (normalized exact match to the configured literal, not word overlap). TP: irrelevant/refused; FP: answerable/refused; FN: irrelevant/answered; TN: answerable/answered. Report precision, recall, F1, false refusal rate FP/answerable, and false answer rate FN/irrelevant, with explicit null/zero-denominator behavior. Keep old `refusal_f1` key as the new binary F1 and label its semantic change in the report.
- Recompute Recall@1/4/8 and MRR from ranked atom IDs and source-owned gold IDs. Only answerable rows contribute. Record atom count, candidate count, evaluated K, and saturation (`min(K,candidates)/candidates`); do not interpret Recall@8=1 on nine atoms as strong top-rank retrieval. Existing prediction files lacking gold IDs may be joined by question ID with atomized JSONL only after exact question/answers checks and source validation; otherwise report unavailable, never infer hits from the old `recall_at_k`.
- Generation output rows include gold atom IDs, atom/candidate counts, checkpoint step, config and dataset fingerprints, and settings version. Metrics and a human-readable machine-readable evaluation report are written under the run's `metrics/` and `reports/`; prediction JSONL defaults to `predictions/`, and the notebook points there. Existing explicit `--output` remains supported; no silent moves/deletes.
- A local comparison command/report accepts named prediction files under the same question set, source gold, split, dataset fingerprint and evaluation protocol; reject incompatible settings or identify legacy rows as non-comparable. Do not present step-28 vs step-31 as a controlled before/after. W&B logging of evaluation outputs is opt-in and tested offline; it must not upload in this task.

## Checkpoint artifact contract

- Preserve local best/last retention and exact-resume data. Add an explicit policy equivalent to `best_weights_and_last_full`; current `best_weights`/`none` retain existing behavior for old configs and checkpoints. Upload a verified last checkpoint at a terminal/early-stopped run only, with config and run manifest, under stable `last` alias and a local marker to avoid duplicate uploads on resume. Best remains model-only. Optional periodic last upload may be configured separately, disabled by default; a hard process kill between periodic uploads is not recoverable from W&B.
- For ZeRO-2, last artifact must contain all committed files needed by the supported distributed resume loader, not only portable weights. Never include atomization raw provider responses. Upload failure must not corrupt local checkpoint pointers; report it explicitly.
- Online W&B upload is not marked complete until the artifact upload reports completion. Offline mode records an `offline_pending` marker, explicitly not remote durability; a later W&B sync must be reported separately. If artifact upload fails, keep the locally committed best/last checkpoints and finish training with a structured upload-failure warning/status rather than misreporting the checkpoint as uploaded or failing the already completed training computation.
- An existing checkpoint trained under a different config is not used to score the supplied step-28 predictions. Its fingerprint is verified in local smoke/read-only checks.

## Baseline comparison boundary

Baseline-specific prompt/context construction remains distinct by definition. The shared generation protocol requires greedy decoding, the same one-line stopping rule and cap, but does not require byte-identical prompts for Doc-to-LoRA versus Doc-to-Atom. Comparison enforces identical question IDs, immutable source gold, split, dataset fingerprint, normalization/scoring version and stopping/decoding settings; it reports each baseline's prompt ID separately. A mismatch in these shared fields is a hard error, not a silently averaged score.

## Testing and reporting

- TDD: unit tests for normalization/multi-reference, first-line diagnostic, refusal confusion matrix and zero denominators, retrieval cutoffs/saturation, generation stop including leading newline and combined token, path and metadata contracts, W&B artifact contents and policy. Integration tests cover score-only enrichment, checkpoint mismatch rejection and artifact logging with a fake/offline W&B client.
- Local smoke exercises toy training, committed checkpoint save/load, and evaluation generation/scoring on a tiny synthetic record without paid API or full model download. Inspect produced JSONL, metrics and report. Run the full relevant test suite.
- Report old-file metrics and new scorer metrics on the exact same immutable predictions, separating scoring changes from generation changes. Real before/after inference on the supplied run remains unavailable without its step-28 checkpoint and Gemma model locally. Explain empty `memory/` as an ephemeral cache, not fabricate files to fill directories.
