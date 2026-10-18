#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
BLOCKED: the ZeRO-2 launcher is intentionally fail-fast.
Task 5 CPU/Gloo checkpoint/resume equivalence is complete, but the required Molab
2-GPU CUDA/NCCL/DeepSpeed ZeRO-2 evidence has not been validated on this checkout.

Run on Molab:
  uv run pytest tests/test_molab_zero2.py -q -m molab --run-molab

Keep this launcher blocked until reports/zero2-equivalence.json says PASS and the
checkpoint audit confirms at least one ZeRO optimizer-state shard per rank. A
rank-0 optimizer.bin alone is not acceptable ZeRO-2 resume evidence.
EOF
exit 2

