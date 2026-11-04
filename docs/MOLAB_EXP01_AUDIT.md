# Audit: Molab SQuAD exp01

Evidence root: `outputs/molab-unified/squad_molab/four_stage/exp01/`.
The run, predictions, resolved config and checkpoint metadata were only read.
No model training, model generation, Molab execution or checkpoint rewriting was performed.

## What the saved evidence shows

- All 800 optimizer updates completed: 200 per curriculum stage. The report has
  no non-finite loss entries; optimizer learning rates/gradient norms are finite.
- `best.json` points to step 608, `last.json` to step 800. Both pointers are left
  unchanged. Step 608 is also the best *observed stage-4* checkpoint under source
  answerable LM, so changing selection metric does not improve this old run.
- At step 608, source-answerable validation LM is **5.2299**, whereas generated
  refusal LM is **0.9973**. The overall **2.4129** averages 100 source questions
  and 199 refusal probes. It is not evidence of strong source-answer quality.
- At step 800, train answerable LM is **0.5332**, but validation answerable LM is
  **5.7742**. This large gap and worsening stage-4 validation suggest poor
  generalization; more steps alone are not a demonstrated remedy.
- The score-only generation evaluation remains F1 **6.97%**, EM **3.00%**,
  false-refusal rate **51%**, and gold recall@8 **95.50%**. Retrieval recall is
  not answer accuracy, especially with many small atom banks.
- Warmup is **500 / 800 updates (62.5%)**: it extends past the first two shortened
  curriculum stages. This is a schedule choice, not a runtime crash.

## Confirmed code inconsistencies

1. Legacy training prompts are `Question: ... Answer:`, while generation uses
   extra short-answer and refusal instructions. The model is optimized under a
   different textual policy from the one used for saved generation.
2. Training targets contain source answer text but no explicit end-of-answer
   newline, whereas generation stops after a completed answer line.
3. Loop KL uses `batchmean` on `[1, answer_tokens, vocabulary]`: it sums token
   divergences per sequence. The shared masked distillation helper averages over
   answer tokens. Thus relative objective weighting differs with answer length.
   Composition KL has the same dimensional-reduction issue.
4. One source question plus two auxiliary probes per record receives a 1:2
   question-count weighting. Mixed validation LM likewise emphasizes refusal.
   This is an observed imbalance, not proof that it alone caused poor F1.

## Implemented fixes for a new run

`configs/experiments/molab_aligned_qa_v2.yaml` explicitly opts into:

- `objective_protocol: aligned_qa_v2`: shared train/generation prompts, supervised
  answer-line termination, FP32 LM/KL computation, token-mean distillation and
  composition KL, and configured routing class weights.
- `question_loss_reduction: group_balanced`: equal total influence for the
  available source-answerable and refusal groups in each record.
- `early_stopping_metric: answerable.losses.lm`.
- A 40-update warmup for this 800-update diagnostic.

Missing `objective_protocol` means **legacy_v1**; missing question reduction means
the original question mean. This preserves old run behavior and exact-resume
compatibility. The new explicit config has a different fingerprint and must not
be used to resume exp01. No source questions or answers are modified.

The notebook exposes **4-stage aligned QA v2 (NEW run)** in a separate output
namespace. Select a new Run ID when starting it. Keep the original diagnostic
profile when inspecting or resuming exp01. Nothing has been launched automatically.

Training console output is now compact. Full tokenwise teacher-forced examples
remain in `reports/training_result.json`; use `--full-report` to print them all.
Trace summaries label their prediction mode so they cannot be confused with
autoregressive evaluation. These display changes do not alter losses/checkpoints.

## Limits and remaining safeguards

These code fixes and configuration choices pass local CPU regression tests.
**Their effect on F1 is unmeasured until a separately authorized new run.** No
improved model or new benchmark result is claimed.

The old checkpoint states contain empty dataset/prompt fingerprint fields.
Their original config fingerprint is preserved and verified, but absent historic
dataset bindings cannot be reconstructed by editing metadata. Do not replace old
checkpoint fingerprints or resume on different data to bypass identity checks.

Read-only machine audit:

```powershell
uv run scripts/audit_molab_run.py `
  --run outputs/molab-unified/squad_molab/four_stage/exp01 `
  --output outputs/molab-exp01-audit-20261104-v2.json
```

The audit refuses to overwrite reports or write inside the historical run. It
records evidence hashes, stage counts, validation history, selected checkpoint,
metrics and findings without loading or executing model weights.
