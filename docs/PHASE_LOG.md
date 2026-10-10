# Phase ledger

1. **Paper audit and isolated repository — complete.** Added `PAPER_SPEC.md`,
   `REPRODUCTION_NOTES.md`, configurations, packaging, and an independent Git
   root. The upstream repository had no code to port.
2. **Atomization — implementation complete; live semantic run blocked.** Added
   schemas, reconstructed prompts, MiniMax client, tolerant parsing, deterministic
   cross-field validation, and CLI. Prompt literals and decoding values are
   marked assumptions. The environment has neither `MINIMAX_API_KEY` nor
   `MINIMAX_BASE_URL`, so no fabricated replacement was used.
3. **Compiler — complete.** Added shared early-exit encoder, normalized key,
   rank-8 factors, zero-B initialization, sparse logits, and memory bank. Direct
   parallel factor heads are an explicit assumption because head factorization is
   unpublished.
4. **Router/composer — complete.** Added cosine K1 retrieval, actual frozen BGE
   integration, metadata features, top-K softmax, Equation (1), and one gate per
   adapter target. Metadata encoding and rerank loss form are explicit assumptions.
5. **Injection — complete.** Dynamic adapters target q/v/o/down in the last four
   layers and leave the base frozen.
6. **Objectives/curriculum — complete.** Implemented all disclosed objective
   families and the four curriculum stages. Loss coefficients, `rho`, and several
   exact objective forms are unpublished and live only in the fallback config.
7. **Training/inference/evaluation — complete.** Added teacher/student execution,
   accumulation, cosine schedule, clipping, checkpoint/resume, offline/online
   separation, baselines, metrics, and per-example traces.
8. **Synthetic smoke and unit tests — complete.** The smoke performs forward,
   all losses, backward, optimizer step, and checkpoint round-trip.
9. **Real-data subset — partially complete.** Downloaded and grouped a small
   SQuAD subset into `data/squad_small_raw.jsonl` (3 documents from 8 train and 4
   validation rows). Atomization is blocked only on the missing MiniMax endpoint
   credentials.
10. **Full reproduction run — intentionally not started.** The paper-scale run
    requires gated Gemma access, MiniMax credentials, unpublished coefficients,
    and the disclosed 8 x A100 80GB profile. The repository fails fast rather
    than claiming equivalence under guessed settings.
