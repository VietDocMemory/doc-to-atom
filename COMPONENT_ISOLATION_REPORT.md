# Component isolation report

## Conclusion

**Primary bottleneck: COMPILER / PARAMETRIC-MEMORY CONTENT TRANSFER.**

**Secondary bottleneck: DECODING / REFUSAL INTERFERENCE.**

The router is not the principal failure, and the audited atomization/gold mappings are sufficient
for these six held-out examples. Forced-gold routing changes validation/test word F1 by only
0.000/+0.019 versus predicted routing. In contrast, putting the same gold atom text directly in
the prompt improves word F1 over forced-gold memory by +0.541/+0.270. Manual inspection finds the
correct answer value in all six gold-text generations (the automated near-match rule counts five;
`80%` versus gold `About 80%` is the conservative miss), but in none of the six forced-memory
generations.

This is a six-example component diagnostic, not a paper-scale result.

## Protocol

- Checkpoint: the unchanged held-out `last.pt`, optimizer step 45. `best.pt` was not selected for
  this experiment.
- Data: the existing document-disjoint validation (3 documents) and test (3 documents) splits,
  one answerable question per document. No data was edited, trained on, or used for tuning.
- Test was evaluated only after validation; the evaluation script performs no optimization.
- Model: frozen `google/gemma-2-2b-it`; greedy decoding, at most 16 new tokens.
- Predicted routing uses the existing top-8 route.
- Forced-gold routing selects exactly `gold_atom_ids`. For `n` gold atoms, every weight is `1/n`,
  so weights sum to 1. All six examples here have one gold atom and therefore weight 1.0.
- Exact match compares normalized complete strings. Near match is a non-exact output containing
  the normalized gold answer, or word F1 >= 0.8. This preserves a separate exact score while
  recognizing an answer followed by an explanation.
- Teacher-forced NLL is mean token NLL for the gold-answer continuation. KL is
  `KL(full-document teacher || mode)` at temperature 1.5, matching the training implementation
  without `T²` rescaling. The tiny negative full-context KL values are floating-point zero.
- Compiler, router, composer, and dynamic LoRA architecture were not modified. Micro-KV remains
  disabled under the existing resource configuration.

## Aggregate metrics

### Validation: unseen documents

| Mode | Word F1 | ROUGE-L | Exact | Near | Exact/near | NLL | KL | Adapter norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base query-only | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 4.969 | 6.075 | n/a |
| Base + full document | **0.778** | **0.778** | 0.667 | 0.333 | **1.000** | 2.748 | 0.000 | n/a |
| Doc2Atom predicted | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3.446 | 6.742 | 0.514 |
| Doc2Atom forced gold | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 3.410 | 6.604 | 0.666 |
| Base + gold atom text | 0.541 | 0.541 | 0.333 | 0.667 | **1.000** | **2.388** | **0.588** | n/a |

### Test: unseen documents

| Mode | Word F1 | ROUGE-L | Exact | Near | Exact/near | NLL | KL | Adapter norm |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Base query-only | 0.194 | 0.194 | 0.000 | 0.000 | 0.000 | 5.546 | 6.386 | n/a |
| Base + full document | **0.607** | **0.607** | 0.333 | 0.333 | 0.667 | 4.633 | 0.000 | n/a |
| Doc2Atom predicted | 0.048 | 0.048 | 0.000 | 0.000 | 0.000 | **3.838** | 6.228 | 0.536 |
| Doc2Atom forced gold | 0.067 | 0.067 | 0.000 | 0.000 | 0.000 | 3.861 | 6.519 | 0.812 |
| Base + gold atom text | 0.337 | 0.337 | 0.000 | 0.667 | 0.667 | 4.142 | **0.999** | n/a |

The test NLL ordering differs from greedy quality: both Doc2Atom modes assign the literal gold
continuation more probability than text-context baselines but still decode incorrect sequences.
This is important evidence for decoding interference, not evidence that forced memory successfully
communicates the answer. KL remains much closer to the full-document teacher for gold-text context
(0.999) than forced memory (6.519).

## Per-example diagnostics

`E` means exact, `N` means near, and `-` means neither. Base modes select no memory atoms; the
gold-text baseline lists the atom inserted as text rather than a composed adapter.

### Validation

#### Scholastic Magazine — gold `September 1876`, gold atom `atom_7`

| Mode | Prediction | F1 | NLL | KL | Norm | Match | Selected atoms |
|---|---|---:|---:|---:|---:|---|---|
| Query-only | `...began publishing in 1883.` | 0.000 | 3.513 | 8.052 | n/a | - | — |
| Full document | `September 1876` | 1.000 | 1.307 | 0.000 | n/a | E | — |
| Predicted | `unanswerable...` | 0.000 | 3.145 | 8.937 | 0.562 | - | `atom_7, atom_12, atom_8, atom_6, atom_17, atom_10, atom_9, atom_21` |
| Forced gold | `1923 ... This is a correct answer.` | 0.000 | 3.175 | 8.698 | 0.702 | - | `atom_7` (1.0) |
| Gold atom text | `September 1876 ... Explanation` | 0.400 | 1.211 | 0.705 | n/a | N | `atom_7` as text |

#### Forbes ranking — gold `8th`, gold atom `atom_4`

| Mode | Prediction | F1 | NLL | KL | Norm | Match | Selected atoms |
|---|---|---:|---:|---:|---:|---|---|
| Query-only | `...ranked ... #1...` | 0.000 | 6.860 | 2.067 | n/a | - | — |
| Full document | `8th` | 1.000 | 4.974 | 0.000 | n/a | E | — |
| Predicted | `unanswerable ...` | 0.000 | 4.312 | 3.089 | 0.544 | - | `atom_5, atom_4, atom_7, atom_8, atom_10, atom_9, atom_11, atom_3` |
| Forced gold | `unanswerable ...` | 0.000 | 4.179 | 2.894 | 0.681 | - | `atom_4` (1.0) |
| Gold atom text | `8th` | 1.000 | 4.167 | 0.334 | n/a | E | `atom_4` as text |

#### Albert Zahm — gold `1882`, gold atom `atom_1`

| Mode | Prediction | F1 | NLL | KL | Norm | Match | Selected atoms |
|---|---|---:|---:|---:|---:|---|---|
| Query-only | `...in 19...` | 0.000 | 4.533 | 8.107 | n/a | - | — |
| Full document | `1882 ... Explanation` | 0.333 | 1.962 | 0.000 | n/a | N | — |
| Predicted | `1929 ... This is a trick question.` | 0.000 | 2.882 | 8.199 | 0.435 | - | `atom_1, atom_2, atom_3, atom_4, atom_5, atom_6` |
| Forced gold | `1929 ... This is a trick question.` | 0.000 | 2.876 | 8.220 | 0.615 | - | `atom_1` (1.0) |
| Gold atom text | `1882 ... Explanation` | 0.222 | 1.788 | 0.725 | n/a | N | `atom_1` as text |

### Test

#### Holy Cross headquarters — gold `Rome`, gold atom `atom_2`

| Mode | Prediction | F1 | NLL | KL | Norm | Match | Selected atoms |
|---|---|---:|---:|---:|---:|---|---|
| Query-only | `...is in Notre Dame...` | 0.000 | 7.717 | 2.053 | n/a | - | — |
| Full document | `The headquarters ... are in Rome.` | 0.154 | 9.734 | 0.000 | n/a | N | — |
| Predicted | `Washington, D.C. ... unanswerable` | 0.000 | 4.735 | 1.610 | 0.543 | - | `atom_2, atom_8, atom_9, atom_5, atom_7, atom_11, atom_12, atom_10` |
| Forced gold | `Washington, D.C. ... unanswerable` | 0.000 | 4.566 | 1.722 | 0.756 | - | `atom_2` (1.0) |
| Gold atom text | `Rome. Is this correct?` | 0.400 | 7.973 | 0.848 | n/a | N | `atom_2` as text |

#### Lobund Institute — gold `The Lobund Institute`, gold atom `atom_1`

| Mode | Prediction | F1 | NLL | KL | Norm | Match | Selected atoms |
|---|---|---:|---:|---:|---:|---|---|
| Query-only | `The Notre Dame Institute for Advanced Study...` | 0.333 | 6.323 | 12.457 | n/a | - | — |
| Full document | `The Lobund Institute` | 1.000 | 1.877 | 0.000 | n/a | E | — |
| Predicted | `The C.S.C.S. ...` | 0.143 | 4.639 | 12.256 | 0.551 | - | `atom_7, atom_11, atom_6, atom_2, atom_8, atom_12, atom_10, atom_1` |
| Forced gold | `The C.S.C.S. ... unanswerable` | 0.200 | 4.537 | 12.370 | 0.746 | - | `atom_1` (1.0) |
| Gold atom text | `The Lobund Institute ... Explanation` | 0.429 | 2.109 | 0.994 | n/a | N | `atom_1` as text |

#### On-campus undergraduates — gold `About 80%`, gold atom `atom_1`

| Mode | Prediction | F1 | NLL | KL | Norm | Match | Selected atoms |
|---|---|---:|---:|---:|---:|---|---|
| Query-only | `Approximately 80% ...` | 0.250 | 2.597 | 4.649 | n/a | - | — |
| Full document | `80%` | 0.667 | 2.286 | 0.000 | n/a | - | — |
| Predicted | `unanswerable...` | 0.000 | 2.141 | 4.819 | 0.516 | - | `atom_1, atom_2, atom_9, atom_12, atom_4, atom_6, atom_7, atom_8` |
| Forced gold | `unanswerable...` | 0.000 | 2.482 | 5.465 | 0.934 | - | `atom_1` (1.0) |
| Gold atom text | `80% ... Explanation` | 0.182 | 2.345 | 1.155 | n/a | -¹ | `atom_1` as text |

¹ Conservative automated matching does not equate `80%` with `About 80%`; manual inspection
considers the answer value correct.

## Bottleneck decisions

### Router bottleneck? No, not primary

- Routing gap (forced minus predicted F1): validation 0.000, test +0.019.
- Routing-gap NLL (predicted minus forced): validation +0.036, test -0.023.
- Every held-out gold atom was already inside predicted top-8. Removing distractors and assigning
  the gold atom weight 1.0 does not recover generation.
- Forced routing actually raises mean adapter norm from 0.514 to 0.666 on validation and from
  0.536 to 0.812 on test, without making answers correct. The issue is not simply insufficient
  adapter magnitude.

### Atomization / gold mapping bottleneck? No for this slice

- Full-document context answers strongly, and gold-atom text exposes the correct answer value on
  all six examples by manual review.
- Gold-text KL is far below forced-memory KL on both splits: 0.588 versus 6.604 on validation and
  0.999 versus 6.519 on test.
- Therefore the selected gold atoms contain usable evidence and are mapped to the right questions.
  This does not certify the entire cache, only the evaluated held-out slice.

### Compiler / memory bottleneck? Yes, primary

- Parametric-memory F1 gap (gold text minus forced memory): validation +0.541, test +0.270.
- Forced-gold memory produces zero exact or near matches on both splits, while gold text produces
  1.000 validation and 0.667 automated test exact/near rates.
- Even with the only gold adapter selected at weight 1.0, the output frequently substitutes an
  unrelated year, institution, or location. The compiler/composer/dynamic-LoRA pathway is not
  reliably preserving the atom's answer-bearing content under this short training regime.
- NLL supports this strongly on validation (+1.022 forced-memory penalty versus gold text), but not
  on test (-0.281). The test disagreement means the diagnosis should rely on the combined greedy,
  containment, and KL evidence rather than NLL alone.

### Decoding / refusal interference bottleneck? Yes, secondary and substantial

- Forced memory improves gold-answer NLL over query-only for every individual example, yet greedy
  decoding never contains the gold answer. Some answer probability is transferred, but it does not
  win the autoregressive decoding path.
- Four of six forced-gold generations contain `unanswerable` or a truncated equivalent, although
  every evaluated question is answerable. This matches the held-out report's observation that the
  short balanced training run learned a strong refusal-token prior.
- The remaining forced outputs (`1923`, `1929`) are confidently wrong content, so refusal bias is
  not the sole cause; it compounds weak parametric content transfer.

## Artifacts

- Evaluation code: `scripts/run_component_isolation.py`
- Aggregate results: `outputs/component_isolation_v2/results.json`
- All 30 per-mode predictions, IDs, weights, NLL, KL, and norms:
  `outputs/component_isolation_v2/predictions.jsonl`

No full training was run, no held-out data was used for training, and no architectural component
was changed.
