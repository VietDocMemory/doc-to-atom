# Paper specification audit

Source: *Doc-to-Atom: Learning to Compile and Compose Memory Atoms*, arXiv:2606.12400v1 (10 June 2026), local `docs/2606.12400v1.pdf`, 20 pages including Appendices A–F.
Re-read for the local MiniMax migration; current implementation gaps are in
`docs/MINIMAX_MIGRATION_PAPER_AUDIT.md`. This specification is not a completion claim.

## Clearly specified

- Offline LLM semantic atomization followed by offline memory compilation; online query routing and composition.
- MiniMax-M2.5 annotations with structured XML and three stages: decomposition, irrelevant-probe generation, question annotation.
- Five atom types, all fields in Table 4, nine relation types, ten question types, deterministic span and cross-ID validation, and failure policy in Appendix B.
- Shared atom/query encoder: first four frozen base-model layers, attended mean pooling, `LN -> Linear(hidden,256) -> GELU -> Linear(256,256)`.
- Compiler trunk: `LN -> Linear(256,256) -> GELU -> Linear(256,256)`; normalized 256-D provenance key; micro-LoRA; optional micro-KV; per-layer/module sparse-mask logits.
- Rank 8, alpha 16, dropout 0, `A ~ N(0,0.02^2)`, `B=0`; targets `q_proj`, `v_proj`, `o_proj`, `down_proj` in the last four blocks.
- Gemma layers 22–25 and Qwen layers 32–35; base bf16 and frozen, trainable modules fp32.
- Router stage 1 cosine/MIPS over L2-normalized keys with K1=32; frozen `BAAI/bge-reranker-base`, max length 192, fusion weight 1.0; four metadata signals; temperature 1.0.
- Composer equation (Eq. 1), sparse gate, and optional weighted/concatenated micro-KV prefix.
- Teacher full-context vs student query-only forwards and Eqs. 2–8; routing BCE weights 0.5/1.0/1.5; distillation temperature 1.5.
- Four curriculum stages: `[0,2000): K=1`, `[2000,6000): K=4`, `[6000,9000): K=8`, `[9000,+): K=8`, including loss activation order.
- AdamW betas, learning rates, weight decay, cosine schedule, 500 warmup steps, clip 1.0, batch 128, data maxima, greedy 128-token evaluation, up to 50 epochs and validation-LM early stopping.
- Paper runtime: PyTorch 2.1, 8 x A100 80GB, NVLink, DDP with `find_unused_parameters=true`, DeepSpeed ZeRO-2, FlashAttention-2, 8 dataloader workers/rank, prefetch 4, persistent workers.

## PAPER_UNSPECIFIED

- Official artifacts: the author's public repository `https://github.com/xid32/Doc-to-Atom`
  exists (audited commit `0848bda2aaef69b375436e24a30c8469a3f0560c`, 2026-05-27), but contains
  only a two-line README and no implementation, configuration, data, release, tag, or checkpoint.
  No author checkpoint was found on Hugging Face during the 2026-10-10 audit. Consequently the
  repository cannot resolve any of the implementation details below yet.
- Literal MiniMax system/user prompts, dataset adapters, model endpoint/version snapshot, decoding parameters, irrelevant-probe count, batching, retry budget, and calibration procedure.
- Total-loss coefficients, their numeric curriculum transitions, and target sparse density `rho`.
- Whether supporting atoms are positives in Eq. 4 (the equation says “gold mask” while the schema separately defines supporting atoms).
- Reranker and conflict are described as listwise cross-entropy, but their precise
  normalization, reduction and edge-case rules are not published; the confidence
  alignment formula is not given.
- Exact encoding of conflict group and semantic type into the claimed four-dimensional metadata bias.
- Sparse-gate aggregation details (logit vs probability aggregation), whether a gate multiplies both factors or the resulting delta, and any hard/straight-through sparsification.
- Micro-KV slot count, tensor shapes, initialization, target layers, positional handling, and prefix-injection implementation; the main setting does not say whether micro-KV is enabled.
- Exact teacher/student prompt literals, refusal prompt, chat templates, answer masking, padding side, and distillation reduction/scaling.
- Compiler-head factorization/parameter sharing beyond “parallel heads”; checkpoint format and optimizer epsilon.
- Dataset versions, exact train/validation/test construction, mixture weights, sample counts, seeds, atomization cache, and LongBench preprocessing.
- Early-stopping patience/min-delta; per-rank microbatch and gradient accumulation.
- The paper says irrelevant queries yield near-zero routing weights, but also defines final top-K weights by softmax, which always sums to one; it does not specify a null-memory gate.
- Table 3 describes “same six benchmarks” while the main table has five in-domain datasets plus eight LongBench subsets; exact efficiency aggregation is unclear.

Paper configs retain nulls for unpublished loss weights and density. Runnable
assumptions also appear in reconstruction/experiment configs and notebook builders,
not only `resource_constrained.yaml`. Some existing architectural choices are
hard-coded; see the current audit rather than interpreting a config name as proof
of equivalence to the paper.

