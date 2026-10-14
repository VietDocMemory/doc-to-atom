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

## Paper-style benchmark reporting

`scripts/run_paper_benchmark.py` evaluates a trained Doc2Atom checkpoint on
atomized records whose `split` is `test`, then writes scored predictions and
F1/ROUGE-L Markdown tables.  It evaluates `Base+ctx` and `Doc2Atom` directly.
The original Doc-to-LoRA code/checkpoints are not part of this repository, so
`D2L-ckpt`, `D2L-raw`, and `D2L-atom` are supplied as external prediction JSONL files rather
than silently substituted with another implementation.

```bash
uv run scripts/run_paper_benchmark.py \
  --config configs/resource_constrained.yaml \
  --data data/benchmark_atoms.jsonl \
  --checkpoint outputs/train/best.pt \
  --model-label gemma2_2b_it \
  --external-predictions d2l_ckpt=outputs/d2l_ckpt_predictions.jsonl \
  --external-predictions d2l_raw=outputs/d2l_raw_predictions.jsonl \
  --external-predictions d2l_atom=outputs/d2l_atom_predictions.jsonl \
  --output-dir outputs/paper_benchmark/gemma

uv run scripts/summarize_paper_benchmark.py \
  --predictions outputs/paper_benchmark/gemma/scored_predictions.jsonl \
  --predictions outputs/paper_benchmark/qwen/scored_predictions.jsonl \
  --output-dir outputs/paper_benchmark/combined
```

Rows supplied by an external D2L file must have `dataset`, `split`,
`prediction`, and `gold`; `model` is optional because the runner fills it from
`--model-label`.  The shipped benchmark manifest is
`configs/paper_benchmark.yaml`.  Its dataset versions/splits must be pinned by
the experimenter because the paper does not disclose them.
For a longer single-GPU D2A-only run, use `configs/resource_constrained.yaml`
for Gemma or `configs/qwen_resource_constrained.yaml` for Qwen; the
`*_heldout_small.yaml` profiles deliberately cap a small diagnostic.

Prepare each pinned source split before atomization.  This supports local JSONL
exports as well as explicit Hugging Face dataset revisions:

```bash
uv run scripts/prepare_benchmark.py \
  --dataset squad --split train --source-jsonl sources/squad_train.jsonl \
  --output data/raw/squad_train.jsonl
uv run scripts/prepare_benchmark.py \
  --dataset longbench_qasper --preprocessor longbench --split test \
  --hf-dataset THUDM/LongBench --hf-config qasper --hf-split test \
  --hf-revision PINNED_COMMIT --max-rows 100 \
  --output data/raw/longbench_qasper_test.jsonl
```

### D2A-only multi-dataset workflow

`prepare_d2a_benchmark.py` and `run_d2a_benchmark.py` use the same
`--datasets ...` pattern as the adjacent Doc-to-LoRA benchmark scripts.  They
run **only** `Base+ctx` and `Doc2Atom`; D2L is neither run nor reported.
Preparation intentionally stops before atomization, because atomizing a full
suite calls the configured provider and can incur API cost.

```bash
# Download + normalize sources. Add --max-rows 5 for an inexpensive smoke run.
uv run scripts/prepare_d2a_benchmark.py \
  --datasets squad drop ropes longbench_2wikimqa longbench_hotpotqa \
             longbench_multifieldqa_en longbench_multifieldqa_zh \
             longbench_musique longbench_narrativeqa longbench_qasper longbench_triviaqa \
  --split test --output-dir data/benchmark_raw

# Atomize every normalized file. The explicit loop makes API cost visible.
for f in data/benchmark_raw/*_test.jsonl; do
  name=$(basename "$f")
  uv run scripts/atomize.py --provider openai --input "$f" --output "data/benchmark_atoms/$name"
done

# Load the checkpoint once and produce one combined D2A-only table.
uv run scripts/run_d2a_benchmark.py \
  --config configs/resource_constrained.yaml \
  --checkpoint outputs/train/best.pt \
  --datasets squad drop ropes longbench_2wikimqa longbench_hotpotqa \
             longbench_multifieldqa_en longbench_multifieldqa_zh \
             longbench_musique longbench_narrativeqa longbench_qasper longbench_triviaqa \
  --data-dir data/benchmark_atoms --split test --model-label gemma2_2b_it \
  --output-dir outputs/d2a_benchmark/gemma
```

The paper image also lists standalone `2wikimultihopqa` and `qasper`. Their
source versions are not disclosed by the paper, so they are deliberately not
auto-downloaded. Export a pinned release to JSONL, prepare it with
`prepare_benchmark.py --source-jsonl ...`, and atomize it before evaluation.

OpenAI deviation example:

```bash
uv run scripts/atomize.py --provider openai \
  --input data/squad_small_raw.jsonl \
  --output data/squad_small_atoms_openai.jsonl
uv run scripts/train.py --config configs/resource_constrained.yaml \
  --data data/squad_small_atoms_openai.jsonl --max-steps 1 --max-questions 1
```

Credentials are loaded from the ignored `.env`. MiniMax uses `MINIMAX_API_KEY`,
`MINIMAX_BASE_URL`, and optional `MINIMAX_MODEL`. The supported OpenAI deviation uses
`OPENAI_API_KEY` and optional `OPENAI_ANNOTATION_MODEL` (default `gpt-4.1-mini`) with
`--provider openai`. Gemini uses `GEMINI_API_KEY` and optional
`GEMINI_ANNOTATION_MODEL` (default `gemini-3.5-flash-lite`) with `--provider gemini`.
Gemini calls Google's OpenAI-compatible endpoint. Prompts are reconstructed, not
official; non-MiniMax output is tagged `REPRODUCTION_DEVIATION` in annotation provenance.

See `docs/MOLAB.md` for the paper distributed profile and single-GPU fallback.

The upstream repository linked by the paper was audited at commit
`0848bda2aaef69b375436e24a30c8469a3f0560c`; it contained only a two-line
README, so no upstream implementation or checkpoint was available to reuse.

