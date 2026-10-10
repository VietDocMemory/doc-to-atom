# Parametric memory transfer isolation

## Conclusion

On the controlled training example, both direct LoRA and compiler-generated LoRA encode the gold
atom knowledge. The compiler-generated adapter is actually cleaner and much smaller than the
direct oracle. This rules out a fundamental bug in LoRA injection, target selection, scaling,
autograd connectivity, or the compiler's ability to emit useful A/B factors.

The earlier held-out failure is therefore not CASE A or CASE B in their strong forms. It is a
**compiler conditioning/generalization problem under the small training regime**, combined with
**decoding behavior**. The irrelevant objective is not the main cause of answer-content failure on
this train example, although its suppression term can conflict with refusal generation when driven
too strongly.

This is a one-example train-set diagnostic, not a reproduction result.

## Controlled example and protocol

- Split/record: train-only `squad_train_7`; validation and test were never loaded into an optimizer.
- Question: “How many stories tall is the main library at Notre Dame?”
- Gold answer: `14-story`.
- Gold `atom_2`: “The main building is the 14-story Theodore M. Hesburgh Library.”
- Irrelevant probe: “What are the operating hours of the Theodore M. Hesburgh Library?”
- Base: frozen `google/gemma-2-2b-it`.
- Targets: rank 8, alpha 16, `q_proj`, `v_proj`, `o_proj`, and `down_proj` in decoder layers
  22–25 (16 targets total).
- Objective: answer LM + full-context-teacher KL. The mixed compiler run additionally uses
  irrelevant LM + KL + `0.1 * suppression`.
- Main budget: 40 answer updates in every learned mode. Direct LR is 0.01 and compiler LR is the
  resource profile's 0.0001. Different LRs reflect the very different direct-factor/head
  parameterizations; comparisons are capability diagnostics, not optimizer-efficiency claims.
- Router is bypassed. The answer path selects only `atom_2` at weight 1. The irrelevant path uses
  a uniform composition of all document atoms, without predicted routing.
- Greedy decoding uses at most 16 new tokens.

Strict exact compares the entire generated string. Because Gemma often appends an explanation, the
report also records first-line exact and answer containment. Teacher-forced NLL and KL are not
affected by explanation length.

## Transfer ladder

| Mode | Train F1 | NLL | KL | Strict exact | First-line exact | Contains gold |
|---|---:|---:|---:|---:|---:|---:|
| Base query-only | 0.000 | 5.637 | 5.900 | 0 | 0 | 0 |
| Base + gold atom text | 0.167 | 3.533 | 0.576 | 0 | 0 | 0¹ |
| Direct optimized LoRA | 0.571 | 0.908 | 3.090 | 0 | 0 | 1 |
| Compiler LoRA forced-gold, mixed | 0.400 | 0.542 | 0.650 | 0 | **1** | **1** |
| Compiler LoRA forced-gold, answerable-only | 0.400 | **0.521** | **0.623** | 0 | **1** | **1** |

¹ Gold text generates `14 stories`, which is semantically correct but does not exactly contain the
normalized target `14 story` because of pluralization.

Predictions:

- Base query-only: `12 stories tall...`
- Base + gold atom text: `14 stories...`
- Direct LoRA: `4-story    14-story 14-`
- Compiler mixed: `14-story` followed by an explanation.
- Compiler answerable-only: `14-story` followed by an explanation.

The direct oracle lowers NLL by 83.9% relative to query-only and places the gold answer in its
output, proving that the paper target modules and injection path can encode the atom. Its high LR
produces a malformed greedy sequence and a very large adapter norm (194.24), so it is an existence
oracle rather than a desirable adapter.

The answerable-only compiler lowers NLL by 90.8%, reaches first-line exact, and matches the teacher
distribution much better than direct LoRA while using adapter norm 2.67. The compiler can therefore
generate effective micro-LoRA factors for a seen atom when optimized directly for content transfer.

## Learning curves

Loss values are measured before each listed update; NLL, prediction, and adapter norm are measured
after it.

### Direct optimized LoRA

| Step | LM | Distill | NLL | F1 | LoRA norm | Total grad norm | Prediction |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 5.637 | 5.900 | 2.548 | 0.000 | 6.51 | 21.47 | repeated `14story` |
| 10 | 2.800 | 13.198 | 2.943 | 0.182 | 72.29 | 17.53 | repeated `4/story` |
| 20 | 2.562 | 9.760 | 2.499 | 0.000 | 142.32 | 10.93 | repeated `story` |
| 40 | 0.869 | 3.173 | 0.908 | 0.571 | 194.24 | 3.08 | contains `14-story` |

The direct run is unstable at LR 0.01 (NLL briefly rises to 22.23 at step 2), but ultimately learns
the target. This instability and norm growth are optimization characteristics, not an injection
failure.

### Compiler-generated forced-gold, answerable-only

| Step | LM | Distill | NLL | F1 | Adapter norm | Total grad norm | Prediction |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 5.637 | 5.900 | 2.148 | 0.000 | 5.25 | 134.99 | repeated `14story` |
| 10 | 0.979 | 1.411 | 0.658 | 0.400 | 3.34 | 202.68 | `14-story...` |
| 20 | 0.563 | 0.758 | 0.430 | 0.400 | 3.02 | 10.63 | `14-story...` |
| 40 | 0.510 | 0.637 | 0.521 | 0.400 | 2.67 | 4.61 | `14-story...` |

### Compiler-generated forced-gold, mixed answer/refusal

| Step | LM | Distill | NLL | F1 | Adapter norm | Total grad norm | Prediction |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 5.637 | 5.900 | 2.346 | 0.444 | 3.31 | 4170.77 | `14 story tall...` |
| 10 | 0.503 | 1.148 | 0.501 | 0.400 | 0.84 | 448.70 | `14-story...` |
| 20 | 0.543 | 0.792 | 0.588 | 0.400 | 0.76 | 161.87 | `14-story...` |
| 40 | 0.515 | 0.682 | 0.542 | 0.400 | 0.72 | 65.67 | `14-story...` |

Compared with answerable-only, mixed training changes answer NLL by only +0.021 (+4.1%), KL by
+0.028, and does not change first-line correctness. It reduces answer adapter norm by 72.9%.
Thus refusal training regularizes factor magnitude strongly but does not destroy this memorized
answer.

## Refusal control and staged re-introduction

Before refusal training, the irrelevant probe hallucinates opening hours: refusal F1 0, no
`unanswerable`, adapter norm 3.529. The independently trained mixed run outputs `unanswerable...`,
with refusal F1 0.222, first-line exact refusal, and irrelevant adapter norm 0.129—a 96.3% reduction.

Starting from the answerable-only checkpoint, ten answer+irrelevant updates were added at each
successive irrelevant weight:

| Weight | Answer F1 | Answer NLL | Answer norm | Refusal F1 | Irrelevant norm | Contains `unanswerable` |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.400 | 0.521 | 2.665 | 0.000 | 3.529 | 0% |
| 0.25 | 0.400 | 0.592 | 0.894 | 0.222 | 0.506 | 100% |
| 0.50 | 0.400 | 0.647 | 0.734 | 0.000 | 0.296 | 0%² |
| 1.00 | **1.000** | **0.451** | 0.818 | 0.000 | 0.186 | 0% |

² The λ=0.5 output starts with `unavailable`, a refusal-like synonym that the required literal
`unanswerable` metric scores as zero.

Answer transfer remains intact throughout; at λ=1 it becomes strict exact. Refusal behavior is
non-monotonic: moderate suppression first yields the target, but further suppression makes the
irrelevant adapter approach zero and exposes the frozen base model, which hallucinates library
hours. This indicates an objective-level tension between “generate the refusal token” and “suppress
the adapter,” but not the answer-content destruction hypothesized by CASE C.

## Technical verification

- Base Gemma has zero trainable parameters throughout. Teacher forwards run with no active adapter.
- Student forwards and backwards execute inside the adapter activation context. All 16 persistent
  forward hooks remain registered.
- Direct A and B gradients are nonzero on all 16 targets at step 40. At step 1, A gradients are
  exactly zero and B gradients are nonzero (1.27–9.48), which is the expected consequence of
  `B=0`; after the first update gradients reach both factors.
- Compiler output tensors retain autograd. At step 40, A/B output gradients are nonzero on all 16
  targets; compiler-head gradient norm is 2.84. No factors are detached.
- Injection is `output + gate * (B @ A) * (alpha/r)`. Here `alpha/r = 16/8 = 2`. An exact numerical
  unit test now checks the nested linear result, gate, and scale—not merely that output changes.
- Direct final B norms are 4.90–9.03; compiler final B norms are 0.35–0.77. B is not stuck near zero.
- Direct final unscaled `||B@A||` is 21.07–86.69 per target. Compiler final values are
  0.58–2.26, with a far smaller and better-behaved composed adapter.
- Compiler final A element std is 0.0348–0.0354 and B std is 0.0038–0.0056. Direct values are
  0.0536–0.0624 and 0.0541–0.0665 respectively. Outputs are finite and non-degenerate.

### Final per-target factors and gradients

`ΔW` below is the unscaled `||B@A||`; runtime subsequently applies the verified gate and scale 2.
Gradients are measured immediately before the step-40 update.

| Target | Direct A | B | ΔW | ∇A | ∇B | Compiler A | B | ΔW | ∇A | ∇B |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 22:q | 8.199 | 7.716 | 41.346 | 0.0004 | 0.0004 | 4.752 | 0.572 | 0.967 | 0.0376 | 0.0547 |
| 22:v | 8.473 | 5.128 | 33.813 | 0.0144 | 0.0340 | 4.788 | 0.416 | 0.714 | 0.0559 | 0.1037 |
| 22:o | 7.902 | 7.697 | 40.616 | 0.0629 | 0.7978 | 4.526 | 0.766 | 1.260 | 0.0468 | 0.2410 |
| 22:down | 16.490 | 7.778 | 70.516 | 0.3768 | 1.7227 | 9.530 | 0.661 | 2.258 | 0.1145 | 0.4007 |
| 23:q | 8.147 | 7.531 | 42.760 | 0.00003 | 0.0008 | 4.794 | 0.512 | 0.881 | 0.0175 | 0.0395 |
| 23:v | 7.998 | 5.242 | 28.213 | 0.0102 | 0.0607 | 4.743 | 0.349 | 0.589 | 0.0165 | 0.1119 |
| 23:o | 7.718 | 7.799 | 40.362 | 0.0880 | 0.5421 | 4.460 | 0.551 | 0.868 | 0.0328 | 0.0959 |
| 23:down | 15.807 | 7.868 | 69.728 | 0.1737 | 0.8112 | 9.501 | 0.640 | 2.164 | 0.0974 | 0.2986 |
| 24:q | 8.079 | 7.479 | 38.443 | 0.0115 | 0.0295 | 4.740 | 0.507 | 0.850 | 0.0179 | 0.0373 |
| 24:v | 7.274 | 4.896 | 21.068 | 0.0111 | 0.0280 | 4.737 | 0.347 | 0.582 | 0.0393 | 0.1541 |
| 24:o | 7.129 | 8.124 | 31.468 | 0.0546 | 0.6266 | 4.467 | 0.606 | 0.958 | 0.0393 | 0.1666 |
| 24:down | 15.772 | 7.971 | 70.878 | 1.2746 | 1.3653 | 9.540 | 0.654 | 2.237 | 0.2411 | 0.3705 |
| 25:q | 7.799 | 7.334 | 42.683 | 0.0008 | 0.0162 | 4.766 | 0.618 | 1.063 | 0.0334 | 0.0813 |
| 25:v | 7.738 | 5.082 | 24.827 | 0.0077 | 0.0227 | 4.772 | 0.353 | 0.600 | 0.0509 | 0.1254 |
| 25:o | 7.388 | 8.106 | 35.334 | 0.0418 | 0.3868 | 4.499 | 0.577 | 0.920 | 0.0780 | 0.1979 |
| 25:down | 15.736 | 9.033 | 86.689 | 0.5439 | 0.6263 | 9.522 | 0.628 | 2.124 | 0.3650 | 0.6132 |

## Answers to the four questions

1. **Can direct LoRA encode atom knowledge?** Yes. It cuts NLL from 5.637 to 0.908 and generates
   the gold string, although the chosen oracle LR yields a malformed surrounding sequence and
   excessive norm.
2. **Can compiler-generated LoRA reproduce that behavior?** Yes on this seen example, and more
   cleanly: first-line exact, NLL 0.521, KL 0.623, norm 2.665. It does not establish unseen-document
   generalization.
3. **How harmful is the refusal objective?** In the independent 40-step mixed control, answer NLL
   worsens only 4.1%, exact/containment is unchanged, and refusal changes from hallucination to the
   correct target. It is not the main answer-transfer failure. Excessive sequential suppression
   does make refusal behavior collapse non-monotonically by shrinking memory toward zero.
4. **Where is the current bottleneck?** Not low-level injection, scaling, target modules, or the
   compiler's raw memorization capacity. The evidence points to a combination of **compiler
   generalization/conditioning across atoms and documents**, **decoding**, and a secondary
   **objective calibration conflict** between refusal LM and adapter suppression. More train-only
   multi-document diagnostics are warranted before changing architecture.

Artifacts: `outputs/parametric_transfer_v2/results.json` contains full learning traces and every
per-target distribution/gradient measurement. The executable diagnostic is
`scripts/run_parametric_transfer_isolation.py`.
