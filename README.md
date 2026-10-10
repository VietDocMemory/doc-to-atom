# Doc-to-Atom reproduction

Audit-first reproduction of **Doc-to-Atom: Learning to Compile and Compose Memory Atoms** (arXiv:2606.12400v1). This is an independent project. It does not implement “Doc-to-LoRA per chunk”.

The paper audit is in `PAPER_SPEC.md`; unpublished details and explicit assumptions are in `REPRODUCTION_NOTES.md`. The paper-to-code index and phase ledger are in `docs/IMPLEMENTATION_MAP.md` and `docs/PHASE_LOG.md`.

## Architecture

`document -> MiniMax XML atomization -> shared four-layer encoder -> atom compiler -> memory bank`

`query -> same encoder -> cosine K1=32 -> frozen BGE reranker -> top-K -> composer -> last-four-layer micro-LoRA -> frozen LLM`

Offline memory build and online query inference are separate APIs in `inference/build_memory.py` and `inference/query.py`.

## Commands

```bash
uv sync --extra test
uv run scripts/prepare_dataset.py --dataset squad --output data/squad_small_raw.jsonl
uv run scripts/atomize.py --input raw.jsonl --output atoms.jsonl
uv run scripts/validate_data.py --input atoms.jsonl
uv run scripts/smoke_train.py --config configs/resource_constrained.yaml
uv run scripts/train.py --config configs/resource_constrained.yaml --data atoms.jsonl
uv run scripts/evaluate.py --predictions outputs/predictions.jsonl
uv run pytest
```

MiniMax uses an OpenAI-compatible endpoint configured through `MINIMAX_API_KEY`, `MINIMAX_BASE_URL`, and optional `MINIMAX_MODEL` (default `MiniMax-M2.5`). Prompts are reconstructed, not official.

See `docs/MOLAB.md` for the paper distributed profile and single-GPU fallback.

The upstream repository linked by the paper was audited at commit
`0848bda2aaef69b375436e24a30c8469a3f0560c`; it contained only a two-line
README, so no upstream implementation or checkpoint was available to reuse.

