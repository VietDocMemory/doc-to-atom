# Phase ledger

1. **Paper audit and isolated repository — complete.** Added `PAPER_SPEC.md`,
   `REPRODUCTION_NOTES.md`, configurations, packaging, and an independent Git
   root. The upstream repository had no code to port.
2. **Atomization — complete on the real subset with a declared deviation.** Added
   schemas, reconstructed prompts, MiniMax/OpenAI providers, tolerant parsing,
   deterministic cross-field validation, and CLI. OpenAI
   `gpt-4.1-mini-2025-04-14` replaced MiniMax-M2.5 and is recorded in every
   record's provenance. Output: 3 documents, 28 atoms, 18 questions, zero invalid
   spans. Nineteen deterministic abstraction-level repairs remain as warnings.
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
9. **Real-data subset — smoke complete.** Grouped a small SQuAD subset, atomized
   it to `data/squad_small_atoms_openai.jsonl`, validated it, manually checked QA
   mappings, loaded gated Gemma with `HF_TOKEN`, and completed one real optimizer
   step plus validation and checkpoint under `outputs/real_data_smoke/`.
10. **Full reproduction run — intentionally not started.** The paper-scale run
    still requires the disclosed 8 x A100 80GB profile and unpublished
    coefficients. The completed run is a resource-constrained smoke, not a paper
    metric reproduction.
