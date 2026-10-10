# Reproduction notes

## Scope and status

This repository implements the disclosed Doc2Atom architecture rather than applying Doc-to-LoRA independently to chunks. It does not modify or depend on `trustmargin-experiments`. Doc-to-LoRA is only an optional evaluation baseline.

The official GitHub repository currently contains no code beyond its README. The audit records its
exact commit in `PAPER_SPEC.md`; this reproduction should be re-audited when the authors publish code.

## Mandatory assumptions

1. Prompts under `atomization/prompts.py` are reconstructed from Appendix B and explicitly label themselves `RECONSTRUCTED_PROMPT_NOT_OFFICIAL`.
2. Metadata bias is a trainable linear combination of four scalar features. Semantic type is represented by a learned scalar embedding and conflict membership by a learned binary scalar; the paper does not disclose this encoding.
3. Supporting atoms are treated as routing positives by default because `relevant = gold union supporting`; this can be disabled to follow the narrower wording of Eq. 4.
4. Sparse mask logits are routing-weight averaged, passed through sigmoid, and applied once to the adapter output. This preserves a gate interpretation without accidentally squaring the gate through both factors.
5. `L_confidence` uses mean-squared alignment between selected routing mass and normalized annotation confidence. `L_conflict` uses per-group listwise cross-entropy. Both exact formulas are unpublished.
6. Micro-KV is represented by a strict interface but disabled in paper configs because its disclosed shapes and main-experiment status are insufficient.
7. No null-memory gate is added to the paper path. Irrelevant suppression follows Eq. 5; this exposes rather than hides the paper's softmax/near-zero inconsistency.

## Resource-constrained deviations

- Single-GPU execution, smaller effective batch, gradient accumulation and checkpointing are runtime deviations, never silent algorithm changes.
- FlashAttention-2 may fall back to SDPA when unavailable; each fallback is logged.
- Synthetic smoke tests use tiny stand-ins and do not claim reproduction metrics.
- Full Gemma/Qwen training requires gated model access and substantially more memory than a consumer GPU. The paper reports 13.43GB *additional compile memory* on Gemma, not total training memory; full training includes the frozen model, teacher/student activations, compiler heads, optimizer state and routing models.

## Faithfulness boundary

`gemma_paper.yaml`, `qwen_paper.yaml`, the data schemas, compiler tensor shapes, routing stages, composition equation, injection targets, losses, and curriculum preserve disclosed behavior. Any runnable value absent from the paper is isolated in a separate config and emitted in the run manifest.

