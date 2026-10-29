# Paper-to-code implementation map

This is a file index, **not** an implementation-completeness checklist. Current
status, explicit deviations and missing paths are audited in
`MINIMAX_MIGRATION_PAPER_AUDIT.md`.

| Paper component | Implementation |
|---|---|
| Appendix B atom schemas, relations and QA types | `atomization/models.py` |
| MiniMax three-pass annotation and recovery policy | `atomization/client.py`, `parser.py`, `validation.py` |
| Schema-v2 source-owned questions and role-only provider DTO | `atomization/models.py`, `question_join.py` |
| Source-authoritative legacy migration | `atomization/migration.py`, `migration_cli.py` |
| Raw provider-response provenance | `atomization/provenance.py` |
| Dataset-owned answer-grounding policy and semantic acceptance gates | `data/preprocessors.py`, `data/semantic_validation.py` |
| Derived answer/refusal target policy | `data/targets.py` |
| Shared frozen first-four-layer encoder and mean pool | `encoder/shared.py` |
| Projection and compiler heads | `compiler/model.py`, `compiler/memory.py` |
| Stage-1 cosine retrieval, metadata and frozen BGE reranking | `router/model.py`, `router/reranker.py` |
| Equation (1), sparse gates and optional MicroKV interface | `composer/model.py` |
| Last-four-layer q/v/o/down micro-LoRA hooks | `injection/lora.py` |
| Teacher/student execution | `training/system.py` |
| Curriculum, ten objectives and optimization loop | `training/curriculum.py`, `losses/objectives.py`, `training/loop.py` |
| Offline memory-bank build and online query | `inference/build_memory.py`, `inference/query.py` |
| Metrics, baseline interfaces (not bundled D2L training/checkpoints), traces | `evaluation/` |
| Distributed checkpoint/save resume and Molab gate | `training/checkpoint.py`, `training/distributed.py`, `training/molab_gate.py` |

MicroKV is represented at the compiler/composer interface, but training with it
is rejected explicitly because the paper does not publish its shape, injection
site, or whether it is part of the primary model. This prevents a silent
invention from being reported as a reproduction.
