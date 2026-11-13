# SQuAD GPU diagnostic — 2026-11-13

## Completed changes
- Shared train/inference query encoding: single-question batch, padding/truncation, 256-token cap.
- D2L wrapper supports explicit raw_v1/chat_v1; chat avoids duplicate special tokens and resets adapters on success/failure.
- Train/evaluation CLI consumes the same canonical representation as validation, without rewriting raw archives.
- Selected regression suite: 149 tests passed.

## Input and limits
Source: data/squad_molab_atoms.jsonl plus data/squad_molab_raw.jsonl.
All 600 records pass source-aware validation after deterministic canonicalization: 500 train / 100 validation, zero invalid spans/source-answer mismatches.
The historical file contains 11 generated probes labeled answerable. A separate snapshot applies Appendix-B irrelevance rules and logs transformations. This does not certify synthetic-question semantics. Original source SHA-256 values are unchanged.

Run: outputs/squad-minimax-e2e-20261113/
- Seed 20261113; 8 train documents, 4 validation documents.
- Validation: 4 source-answerable questions and 8 generated irrelevant probes.
- Frozen google/gemma-2-2b-it BF16, FP32 memory modules; BGE reranker on CPU.
- One RTX PRO 6000 Blackwell GPU, SDPA fallback.
- One smoke optimizer step, then restore initial trainable weights.
- Fresh 16-step run; four updates per stage, boundaries 4/8/12, Top-K 1/4/8/8.
- Effective batch 1; context 1024, atom 96, answer 64 tokens.
- Greedy generation, at most 32 new tokens; W&B disabled.
No annotation API, full-data training, or official D2L checkpoint benchmark was run.

## Results
Smoke passed. All 16 diagnostic steps completed; best checkpoint = 16.
Training portion: 37.8 seconds (excludes loading/smoke/generation). Peak allocated VRAM: 11.9 GiB.

| System | Answer F1 | EM | Refusal F1 |
|---|---:|---:|---:|
| Doc-to-Atom | 0.000 | 0.000 | 0.706 |
| Frozen base, query only | 0.000 | 0.000 | 0.706 |
| Frozen base, full context | 1.000 | 1.000 | 0.933 |

Answer F1/EM uses only four source questions; refusal scores include eight synthetic probes. This is not a full SQuAD/SQuAD-v2 benchmark.
Teacher-forced validation answerable LM loss at steps 4/8/12/16: 4.3135 / 4.3264 / 4.2127 / 4.1895. Do not confuse this with generated-answer quality.
Routing recall@1=.75 and recall@8=1.0, but Top-K 8 covers ~92.2% of available atoms. Adapter norm is nonzero; generated QA did not improve over query-only.

## Artifacts
- summary.json: generated-answer metrics, model revision, GPU summary.
- data/canonical_atoms.jsonl: all 600 canonical records.
- data/subset_atoms.jsonl and data/subset_source.jsonl: selected pair.
- data/selection.json: fixed selected IDs, hashes and validation.
- configs/: resolved configurations.
- smoke/reports/training_result.json: one-step result.
- diagnostic/train_log.jsonl: per-step/stage and teacher-forced diagnostics.
- diagnostic/checkpoints/best.json: best-checkpoint pointer.
- evaluation/: real generated predictions for all three systems.
- diagnostic_manifest.json: source/code hashes at execution time.

Reproduce in a NEW output root:
    uv run python scripts/run_squad_diagnostic.py --output-root outputs/squad-minimax-e2e-NEW

Existing diagnostic manifests are refused. The canonical-reader CLI hardening was added after the GPU run; the run already explicitly used a canonicalized snapshot, so it did not consume historical raw probe labels. That reader change was subsequently regression-tested.

## Interpretation
The pipeline is operational on real GPU hardware: loading, forward/backward, optimizer, all shortened stages, checkpoints and generation. It does not yet demonstrate useful memory QA.
Next: establish single-example/tiny-set generation overfit, then train/evaluate on the full 500/100 split with multiple seeds.

The metadata feature mapping and composition equation were not redesigned without exact paper specifications. Official D2L evaluation still needs its own checkpoint/environment; these two baselines are Gemma, not Doc-to-LoRA.
