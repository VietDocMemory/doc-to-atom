# Reproduction notes

## Scope and status

This repository reimplements Doc2Atom. See
`docs/MINIMAX_MIGRATION_PAPER_AUDIT.md` for the current component-by-component audit.
Doc-to-LoRA is only an optional evaluation adapter; its actual checkpoints/training
and benchmark reproduction are not bundled.

The earlier upstream audit found only a README at the commit in `PAPER_SPEC.md`.
That is historical evidence, not a fresh claim about the current upstream state.

## Implementation choices

1. Prompts under `atomization/prompts.py` implement the three-stage XML annotation workflow in Appendix B.
2. Metadata bias is a trainable linear combination of four scalar features. Semantic type is represented by a learned scalar embedding and conflict membership by a learned binary scalar; the paper does not disclose this encoding.
3. The current loop supervises **gold-only** routing masks; supporting masks are
   recorded for diagnostics, not treated as positives. This follows Eq. 4's wording.
4. Sparse mask logits are routing-weight averaged, passed through sigmoid, and applied once to the adapter output. This preserves a gate interpretation without accidentally squaring the gate through both factors.
5. `L_confidence` uses mean-squared alignment between selected routing mass and
   normalized annotation confidence. The paper describes per-group listwise CE
   for conflict, but does not disclose reductions or all corner-case rules.
6. Micro-KV is represented by a strict interface but disabled in paper configs because its disclosed shapes and main-experiment status are insufficient.
7. **Existing architectural/loss deviation:** the router adds a learned null-memory
   candidate and the training loop adds a null-decision term under the route
   coefficient. This changes the top-K normalization and is not specified in the
   paper. It is retained, not hidden or silently removed in the MiniMax migration.
8. New atomization uses MiniMax-M3.1-Flash, not paper MiniMax-M2.5. The default
   completion budget is 8192 including thinking, probe count 2, provider reasoning
   depth `max`; these are not paper settings. Sampling is provider-controlled.

## Resource-constrained deviations

The first two entries below describe **historical research outputs**, not an
available annotation backend. The previous provider implementation has been
removed; its datasets, raw responses and provenance are intentionally preserved.

- The completed real-data smoke used OpenAI `gpt-4.1-mini-2025-04-14` for all three
  annotation passes. Historical outputs retain their provider/model provenance.
- OpenAI emitted scalar QA fields as XML attributes; the tolerant parser accepts
  both attribute and child-element forms. Nineteen atoms used a semantic-type
  label in `abstraction_level`; these known aliases were deterministically
  normalized to `evidence` and retained as warnings. The schema enums were not
  changed.
- Single-GPU execution, smaller effective batch, gradient accumulation and checkpointing are runtime deviations, never silent algorithm changes.
- FlashAttention-2 may fall back to SDPA when unavailable; each fallback is logged.
- Synthetic smoke tests use tiny stand-ins and do not claim reproduction metrics.
- Full Gemma/Qwen training requires gated model access and substantially more memory than a consumer GPU. The paper reports 13.43GB *additional compile memory* on Gemma, not total training memory; full training includes the frozen model, teacher/student activations, compiler heads, optimizer state and routing models.
- The real-data smoke ran one record-step and one question on CPU because the
  environment's PyTorch build has no CUDA support. FlashAttention-2 consequently
  fell back to SDPA. This is a runtime-only deviation recorded in the manifest.

## Faithfulness boundary

`gemma_paper.yaml` and `qwen_paper.yaml` preserve many disclosed hyperparameters and
leave unknown weights null. That is not a guarantee of a paper-faithful executable
path: null routing, gate placement, metadata encoding, loss reductions, data caps,
prompting, and the unwired distributed CLI remain limitations in the current code.
No architecture, training loss, checkpoint format or fingerprint algorithm was
changed during the MiniMax migration.

Earlier small-overfit observations referenced a report no longer present in this
checkout. Their Stage-2 start, zero warmup, duplicated examples, provider retry,
GPU/CPU placement and SDPA fallback are diagnostics, not evidence of paper metrics.

