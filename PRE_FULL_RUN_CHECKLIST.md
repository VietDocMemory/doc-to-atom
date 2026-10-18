# Pre-full-run checklist

Full training and tuning remain prohibited until every gate below has retained evidence.

## Distributed checkpoint/resume

- [x] Single-process versus 2-process CPU/Gloo numerical equivalence.
- [x] Interrupted 2-process CPU/Gloo resume equivalence.
- [x] Exact optimizer/curriculum counters and sampler permutation/cursor restore.
- [x] Per-rank Python, NumPy, and Torch RNG restore.
- [x] Atomic outer `COMMITTED` marker and portable weights.
- [ ] Molab 2-GPU Accelerate/DeepSpeed ZeRO-2 continuous run.
- [ ] Molab 2-GPU interrupted + resume equivalence.
- [ ] At least two `*optim_states.pt` ZeRO shards in the committed checkpoint; a
  rank-0 `optimizer.bin` does not satisfy this gate.

Run the Molab-only gate:

```bash
uv sync --extra test --extra train
uv run pytest tests/test_molab_zero2.py -q -m molab --run-molab
```

Retain the generated `reports/zero2-equivalence.json`, `logs/*.log`, continuous,
interrupted, and resumed checkpoint directories under the run directory. The report
must record config hashes, checkpoint hash, package versions, tolerances, optimizer
shards, and `wandb_mode=offline`.

## Launcher status

- [x] `scripts/molab/train_zero2.sh` remains fail-fast.
- [ ] Remove fail-fast only after the evidence above is reviewed and committed.
- [ ] Do not begin tuning or full training as part of this gate.
