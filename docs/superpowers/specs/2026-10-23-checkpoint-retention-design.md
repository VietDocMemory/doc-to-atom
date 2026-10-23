# Checkpoint retention and W&B upload design

## Intent and constraints

Keep at most two committed full training checkpoints after each successful save: the best validated step and the latest resumable step. Do not change model architecture, losses, optimizer, curriculum, or the v2 checkpoint payload. Preserve exact resume state (including optimizer, scheduler, RNG, sampler and early-stopping state). Never upload `.raw-responses` as part of this training workflow; atomization provenance outside this workflow is unchanged.

## Approaches

- **Chosen:** immutable `step-*` checkpoint directories with atomic `best.json` and `last.json` pointers. Publish a complete checkpoint, validate its committed structure, update pointers, then prune only unreferenced committed step directories. Best and last may reference the same directory. This preserves existing CLI/evaluation pointer behavior.
- Rejected: overwrite fixed `best/` and `last/` directories. Replacing multi-GB directory trees safely across local and ZeRO-2 backends is more complex and risks losing the previous good state.

## Save and retention policy

- A validation improvement saves a full checkpoint and updates both pointers. A checkpoint interval saves and advances `last`; it advances `best` only if validation improved on that step. Validation without improvement and without another save trigger saves nothing.
- End of training, early stop, and an intentional stop at an optimizer-step boundary always leave `last` at the final completed optimizer step. An unexpected process loss resumes from the last committed checkpoint; uncheckpointed steps cannot be recovered exactly.
- Checkpoint writes and pointer publication remain atomic. Before pruning, verify the newly saved directory has its commit marker, required v2 files, and matching optimizer step. If validation, save, or pointer publication fails, retain prior checkpoints and do not prune.
- Pruning only considers direct child directories matching `step-<8+ digits>` under this run's `checkpoints/`. It refuses cleanup when either existing pointer is malformed or points outside that directory, and never touches incomplete directories, unrelated paths, or a referenced checkpoint. In distributed mode, rank 0 owns pointer updates and pruning between collective barriers.
- Existing v2 checkpoints and pointers remain readable. Configuration defaults retain the new bounded policy; explicit config documents retention and upload choices.

## Tracking policy

W&B continues to receive metrics, config, manifests and reports. At a normally completed or early-stopped run, upload **once** the final best `model.safetensors` (or ZeRO-2 `portable/model.safetensors`) as a model artifact containing the resolved config and run manifest. A local upload marker makes a completed-run resume idempotent for the same best step; it is written only after W&B accepts the artifact call. An intentional mid-run stop defers the artifact. This avoids retaining a large remote artifact version per improvement. Do not upload full `training.pt`, Accelerate/DeepSpeed shards, RNG files, or `.raw-responses`. A hard process interruption may occur before this one-time upload; the local `last` remains exact-resume capable. Optional remote backup of full `last` is out of scope and disabled.

## Verification

TDD tests cover trigger scheduling, best/last pointer behavior, retention with a shared target, failed save/invalid pointer safety, W&B artifact contents, exact single-GPU resume after early stop and intentional interruption, and distributed rank-0/barrier behavior. Run the existing full test suite and bounded distributed tests; no full training.
