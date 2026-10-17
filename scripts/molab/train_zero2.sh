#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
BLOCKED: the ZeRO-2 launcher is intentionally fail-fast.
The custom training loop does not yet route model/optimizer/checkpoint state through
Accelerate, so launching multiple ranks would duplicate data and race on artifacts.
Use scripts/molab/train_single_gpu.sh until the distributed checkpoint/resume gate is implemented.
EOF
exit 2

