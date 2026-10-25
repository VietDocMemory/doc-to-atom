# Full-run checkpoint and evaluation guide

The local checkout has **not** run full training. The opt-in upload policy below is for a future new run. Keep the resolved config snapshot paired with its checkpoint: changing `runtime.checkpoint_upload` in an existing run changes the config fingerprint and can invalidate exact resume.

## Checkpoints and W&B artifacts

Set `runtime.checkpoint_upload: best_weights_and_last_full` in a **new** resolved config. Existing `none` and `best_weights` policies retain their behavior. `training.checkpoint_retention: best_last` keeps only the referenced full local `best` and `last` checkpoint directories after each committed save. `last` still contains model, optimizer, scheduler, RNG and trainer state, and is saved at the configured interval and terminal/early-stop points.

At a normal terminal or early-stop boundary, the new policy publishes:

- `RUN_ID-best-model:best`: model weights, resolved config, run manifest. It is for inference, not exact resume.
- `RUN_ID-last-checkpoint:last`: the full committed checkpoint (including ZeRO-2 distributed state where applicable), resolved config and run manifest. It is for exact resume.

Raw provider responses are excluded. No checkpoint artifact is logged on a nonterminal validation step. Artifacts use stable aliases; W&B may retain older artifact versions remotely, so local two-checkpoint retention does not imply only two remote versions. Upload completion is recorded as `uploaded` only after an online W&B upload handle finishes; offline runs record `offline_pending` and require a later `wandb sync`. A failed upload leaves committed local checkpoints and pointers intact and records a warning. Legacy two-field upload markers have unknown remote durability.

For a hard process kill, the most recent interval checkpoint remains the recovery point; the final in-memory updates after it cannot be recovered. An interrupted run should be resumed from `checkpoints/last.json` with the same config and data. The optional full-last upload is terminal/early-stop only, so it does not protect against a Molab instance disappearing before a terminal boundary. To protect that case, use a shorter checkpoint interval and external persistent storage for the local output directory.

## Molab commands after local verification

Run from `/marimo/doc-to-atom` with your existing environment/secrets. Validate source spans before training or evaluation.

```bash
uv run doc2atom-validate --input data/squad_molab_atoms.jsonl --source data/squad_molab_raw.jsonl --reject-span-mismatch
uv run doc2atom-train --config configs/reconstruction/molab_single_gpu.yaml --data data/squad_molab_atoms.jsonl --output outputs/molab-single-gpu --wandb-mode online --wandb-project doc-to-atom
uv run doc2atom-train --config configs/reconstruction/molab_single_gpu.yaml --data data/squad_molab_atoms.jsonl --output outputs/molab-single-gpu --resume latest --wandb-mode online --wandb-project doc-to-atom
uv run doc2atom-evaluate --config outputs/molab-single-gpu/config/resolved.yaml --data data/squad_molab_atoms.jsonl --source data/squad_molab_raw.jsonl --checkpoint outputs/molab-single-gpu/checkpoints/best.json --output outputs/molab-single-gpu/predictions/validation.jsonl --split validation --wandb-mode online
```

The evaluation command refuses to overwrite an existing predictions file. Use a new filename to repeat generation. The CLI writes metrics under `metrics/` and a report under `reports/`. `--wandb-mode` defaults to `disabled` for evaluation. See [local results](docs/evaluation-local-results.md) for the current benchmark and limitations.
