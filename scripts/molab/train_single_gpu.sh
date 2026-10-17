#!/usr/bin/env bash
set -euo pipefail

: "${DOC2ATOM_DATA:?Set DOC2ATOM_DATA to an atomized JSONL file}"
: "${DOC2ATOM_OUTPUT:?Set DOC2ATOM_OUTPUT to a new or resumable run directory}"

config="${DOC2ATOM_CONFIG:-configs/reconstruction/molab_single_gpu.yaml}"
wandb_mode="${WANDB_MODE:-offline}"

test -f "$config" || { echo "missing config: $config" >&2; exit 2; }
test -f "$DOC2ATOM_DATA" || { echo "missing dataset: $DOC2ATOM_DATA" >&2; exit 2; }
mkdir -p "$DOC2ATOM_OUTPUT"
test -w "$DOC2ATOM_OUTPUT" || { echo "output is not writable: $DOC2ATOM_OUTPUT" >&2; exit 2; }

python -c 'import torch; assert torch.cuda.is_available(), "CUDA GPU is required"'

args=(
  --config "$config"
  --data "$DOC2ATOM_DATA"
  --output "$DOC2ATOM_OUTPUT"
  --wandb-mode "$wandb_mode"
)
if [[ "${DOC2ATOM_RESUME:-}" != "" ]]; then
  args+=(--resume "$DOC2ATOM_RESUME")
fi

exec python -m doc_to_atom.training.cli "${args[@]}"

