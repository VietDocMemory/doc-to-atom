# Doc-to-LoRA reference audit

Date: 2026-11-11.

- Doc-to-Atom baseline: `74bf206fc3076e255384e82d6ce30c379893056a`, branch `viet_qwen`.
- Reference checkout: `VietDocMemory/doc-to-lora`, `main`, commit
  `90df54e3b51b552987b700a004ef633b8139a2de`.
- Scope: compare source-level contracts and improve Doc-to-Atom where the reference
  identifies a reproducible numerical issue or a useful safety/compatibility pattern.
  No upstream dependencies, checkpoints, or GPU models were installed or loaded.

## Relationship between the projects

Both projects generate LoRA adapters from frozen-model representations. This conceptual
relationship does **not** make their implementations or checkpoint formats interchangeable.
Doc-to-Atom's core does not import `ctx_to_lora`; cloning the reference next to it does not
wire the two runtimes together. The existing `DocToLoRABaseline` accepts an externally
constructed D2L model; it is not an upstream checkpoint conversion/import mechanism.

| Component | Doc-to-LoRA reference | Doc-to-Atom | Decision |
|---|---|---|---|
| Context representation | `modeling/ctx_encoder.py`: selectable frozen context encoders, including per-layer activations | `encoder/shared.py`: shared early-exit encoder plus trainable projection for atoms and queries | Keep atom/query design; do not truncate the shared model's layer list in place as the reference does for its separate encoder. |
| Adapter generator | `modeling/hypernet.py`: configurable aggregator/hypernetwork | `compiler/model.py`: per-atom keys, A/B heads and sparse gates | Keep atom compiler; upstream checkpoints cannot be loaded directly into these heads. |
| A/B layout | `lora_layer.py`: A is `[r,in]`, B is `[r,out]` | `injection/lora.py`: A is `[r,in]`, B is `[out,r]` | Both implement a valid low-rank delta, using different B layouts; no blind state-dict copy. |
| Injection precision | `lora_forward`: cast activation to A dtype, compute delta, cast final result back to base-output dtype | Previously cast A/B/gate down to activation dtype before multiplying | Fix: preserve native adapter precision, including FP32 factors on a BF16 base; disable autocast only inside delta computation. |
| Pooling | `pooling.py:mean_pool`: mask padded values before reduction | Previously multiply hidden states by the mask, accumulate in hidden dtype | Fix: masked fill before reduction and FP32 accumulation. `NaN * 0` does not remove padded NaNs. |
| Frozen encoder execution | `ctx_encoder.py` and `hypernet.py:generate_weights`: no-grad frozen feature extraction | No explicit no-grad boundary; recursive `.train()` re-enabled frozen base training mode | Harden: no-grad only around frozen feature extraction, keep projection/compiler/router differentiable, and keep the base in eval mode. |
| Composition | `lora_merger.py:combine_lora`: chunk rank concatenation, optional offsets/scalers | `composer/model.py`: query-selected, weighted A/B factors and gates | Keep current composition. Rank growth and sums of chunk deltas are not automatically equivalent to weighted factor composition. No change without a specific equation/experiment. |
| Scale/targets | Reference `model_loading.py` has a different default alpha rule; example main config targets `down_proj` | Atom paper config uses alpha 16/rank 8 and q/v/o/down in last four layers | Keep atom-paper settings rather than copying upstream defaults. |
| Loader compatibility | Reference pins Transformers 4.51.3 and passes `torch_dtype` | Atom allows Transformers >=4.51 but loader used the newer `dtype` spelling | Harden loader to use `torch_dtype`; verified with a legacy-signature stub, not an actual 4.51 installation. |
| Training | Packed contexts/questions, upstream Trainer and task setup | Per-document memory banks, routed multi-objective curriculum | Do not transplant packed adapter assignment: each token/query would need the correct document bank, or packing could leak memory across examples. |
| Prompts | Reference Python API uses model-specific chat templates | Atom has a versioned short-answer prompt protocol | Preserve current protocol in this patch; changing it changes evaluation semantics and needs a separate controlled experiment. |

Paths in the reference column are relative to `src/ctx_to_lora/`; atom paths are relative
to `src/doc_to_atom/`.

## Implemented changes

1. `injection/lora.py`: preserve factor precision; accumulate delta/residual in adapter
   dtype, cast back to the original output dtype. Gate and alpha/rank remain applied once.
   This does not guarantee that every tiny delta survives the final BF16 output rounding.
2. `encoder/shared.py`: mask before pooling, accumulate in FP32, and use `no_grad` only
   for the frozen backbone. `.train()` keeps the backbone in eval mode while leaving
   the trainable projection in training mode.
3. `training/system.py`: recursive `.train()` no longer re-enables dropout/training-mode
   behavior in the shared frozen base. Student gradients through generated adapters are
   retained: eval mode does not disable autograd.
4. `training/factory.py`: use the `torch_dtype` loader keyword compatible with the declared
   lower dependency bound and the reference's pinned Transformers API.

No changes to rank, layer/module targets, compiler shapes, router equations, curriculum,
atomization backend, existing dataset files, or reference repository files.

## Evidence and tests

- On the unmodified atom baseline, the initial regression tests reproduced six failures
  and one passing lifecycle check. In the numerical case, downcasting `[1.001,-1]` before
  multiplying a BF16 `[1,1]` activation erased a representable nonzero adapter contribution.
- Pooling a valid prefix with NaN/Inf padding produced NaNs before the fix.
- Added eight regression cases in `tests/test_doc_to_lora_inheritance.py`, including
  autocast-on/off FP32-factor injection, adapter gradients, padding, frozen/trainable
  execution boundaries, context restoration and the compatible loader signature.
- After fixes: **118 tests passed across 16 selected test files**. These include gradient
  flow, toy training/overfit, inference, checkpoint/resume equivalence and existing
  compiler/router/LoRA tests. This is not the full repository suite or a multi-GPU run.
- No real upstream checkpoint loading, FA2/CUDA performance test, or QA quality benchmark
  was run. Source comparison and toy numerical tests cannot establish answer-quality gains.

## Remaining work / deliberate non-changes

- A real upstream D2L baseline still needs its own environment, official loader/checkpoint,
  and explicit prompt protocol. The current baseline feeds the raw question while the
  reference example uses a chat template; do not label those metrics official-equivalent.
- Keep separate uv environments: upstream pins Transformers 4.51.3, Accelerate 1.6.0,
  DeepSpeed 0.17.1 and vLLM 0.8.5.post1. Installing it into the atom environment could
  replace the active training stack.
- Train routing truncates query embeddings to 256 tokens (`training/loop.py`), while
  `QueryEngine.answer` has no corresponding cap. Long-query parity remains to be specified
  and tested; short-question tests do not establish parity for long inputs.
- The earlier Appendix B–D audit's reconstruction choices still apply, including the
  shortened training profile and router metadata's missing `abstraction_level` feature.
- Memory/checkpoint tensor shapes remain compatible, but pooling and injection arithmetic
  changed. Prefer a fresh run and rebuild memory banks for controlled comparisons. Do not
  claim bitwise exact continuation of a checkpoint created with the old numerical path;
  keep the old code revision when exact historical continuation is required.

The reference checkout is read-only for this task. Nothing is automatically committed,
pushed, regenerated, or trained by these changes.
