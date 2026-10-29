# MiniMax-only migration & audit Doc-to-Atom

Ngày audit: 2026-10-29. Phạm vi: thay annotation backend, cleanup và kiểm chứng
**trên local Windows**, không push GitHub, không chạy Molab, không gọi API trả phí.

## 1. Kết luận

- Annotation mới chỉ dùng MiniMax HTTP API, mặc định **MiniMax-M3.1-Flash**.
  Chỉ đọc `MINIMAX_API_KEY` cho authentication; giữ `HF_TOKEN` để tải base model.
  W&B online vẫn là chức năng tracking tùy chọn, không phải annotation backend.
- Đây là **partial reconstruction**, chưa phải tái hiện đầy đủ hay tái hiện số học
  kết quả paper. M3.1-Flash thay M2.5 là **Deviation**, kể cả khi cùng nhà cung cấp.
- Không thay đổi encoder/compiler/router/composer, loss, curriculum, inference,
  evaluation, checkpoint format hay thuật toán fingerprint để làm đẹp mức độ khớp.
- Dữ liệu cũ được giữ đúng nhãn provider/model; không tái gán provenance thành MiniMax.
  Đối chiếu SHA-256 trước/sau: **51/51 file trong `data/` và `outputs/` giữ nguyên**,
  tổng dung lượng 4,675,027,642 bytes; không thêm/xóa file trong hai thư mục này.
- Bộ test hiện tại: **291 passed, 1 skipped**. Smoke tổng hợp local chạy một
  optimizer step, save/load checkpoint thành công. Không có measured benchmark
  F1/ROUGE của paper hay bằng chứng train Gemma/Qwen đầy đủ từ lần kiểm chứng này.

## 2. Nguồn và phương pháp

Đã đọc toàn bộ **20 trang** `docs/2606.12400v1.pdf`, gồm §3, §4, Appendices A–F,
Eqs. (1)–(8), Tables 1–6 và ví dụ atomization Appendix F.

SHA-256 PDF:
`74ce245b8ecfbc8a2c302d8be400c34a4f2ed8deb38e0bcd6b24fa5a0d5e6769`.

Tài liệu MiniMax chính thức được kiểm tra ngày audit:

1. [Create Chat Completion / OpenAPI](https://platform.minimax.io/docs/api-reference/text-chat.md)
   — server, path, Bearer auth, model enum, request/response schema, provider errors.
2. [Chat Completions compatibility guide](https://platform.minimax.io/docs/api-reference/text-openai-api.md)
   — M3.1-Flash, reasoning depth, output separation, token budget, sampling limitations.

Tên “OpenAI-compatible” ở tài liệu nhà cung cấp mô tả **wire protocol**, không có
nghĩa repo sử dụng GPT hay cần tài khoản OpenAI. Code hiện dùng `httpx`, không SDK
`openai`. Không gọi model để xác nhận availability/quota của một tài khoản cụ thể.
Nhận định về upstream README-only trong các tài liệu cũ là audit lịch sử, không
phải kết quả kiểm tra lại repository tác giả tại ngày hôm nay.

### Quy ước phân loại

- **Implemented:** có đường chạy và kiểm thử cho hành vi được mô tả; không suy ra
  đã tái hiện chất lượng/hiệu suất paper.
- **Partial:** có thành phần nhưng còn thiếu đường tích hợp, coverage hoặc semantics.
- **Missing:** chưa có triển khai/artefact chạy tương ứng.
- **Deviation:** đang khác điều paper công bố.
- **Paper-unspecified:** paper chưa đủ thông tin để xác định một lựa chọn gốc.

Một thành phần có thể mang nhiều nhãn, vì có phần đã triển khai và phần chưa rõ.

## 3. Paper vs Implementation

Các đường dẫn code trong bảng đều tương đối từ `src/doc_to_atom/`.

| Thành phần | Paper, bằng chứng | Implementation hiện tại | Phân loại / khoảng trống |
|---|---|---|---|
| Annotator | §4.1 p7; App B p12: MiniMax-M2.5, structured XML | `atomization/client.py`: MiniMax-M3.1-Flash mặc định, HTTP trực tiếp, ba semantic calls | **Implemented / Deviation**: model đổi theo yêu cầu, prompts không chính thức; chưa chạy canary API thật |
| Atom taxonomy / metadata | §3.1 pp3–4; Table 4 p14: 5 types, 9 relation types, 10 question types | `atomization/models.py`, `prompts.py`, `parser.py`: schemas, enums, adapters và XML parsing | **Implemented / Paper-unspecified**: không có literal prompts/official adapters/calibration; minimality và self-contained phần lớn dựa vào LLM, không phải chứng minh tự động |
| Source grounding / roles | App B.1/B.3 pp13–14: substring spans, drop unknown IDs, union roles, derive conflict | `atomization/validation.py`, `question_join.py`, `data/semantic_validation.py`: source-owned question/answers, ID joins, whitespace span checks, transformations | **Implemented / Partial**: chống thay gold/source tốt; canonicalization spans chạy cuối record, chưa có full validated-span gate trước role call. Source-aware CLI validation cần raw source |
| Irrelevant probes / rescue | App B.2/B.3; App F p19 có generated probe được rescue | `client.py`, `question_join.py`, `data/targets.py`: tạo probes, giữ source answers read-only, rescue không bịa gold answer | **Implemented / Paper-unspecified**: probe count=2 là giả định; rescued answerable probe không có source answer bị loại khỏi LM training |
| Failure handling | App B.4 p14: decompose failure discard; QA failure atom-only; probe failure ignore | Semantic parsing lỗi vẫn giữ policy. HTTP/schema lỗi, `length`, empty/filter/tool response được capture rồi dừng | **Implemented / Deviation**: transport fail-fast nhằm tránh chi phí/reuse output thiếu; không tự retry. Paper không công bố policy retry mạng |
| Shared Memory Encoder | §3.2 p4; Table 6 p20: frozen first 4 layers, attended mean, LN→Linear→GELU→Linear, dim256 | `encoder/shared.py`: hook early exit, chung atom/query encoder, fp32 projection | **Implemented**; ToyLM early-exit/gradient tests không thay thế xác minh mọi model/runtime |
| Compiler trunk/key | §3.2 pp4–5; Table 6: LN/MLP 256, normalized key256 | `compiler/model.py`, `compiler/memory.py`: trunk, key head, independent generated memories | **Implemented**; parallel dense factor-head parameterization là lựa chọn reconstruction |
| Micro-LoRA | Table 6: r8, α16, dropout0, A std0.02/B0; last4 q/v/o/down | `compiler/model.py`, `injection/lora.py`: heads, zero B, target discovery, context-scoped hooks; Gemma22–25/Qwen32–35 configs | **Implemented / Paper-unspecified**: std0.02 áp lên A-head weights, không đảm bảo phân phối emitted A chính xác như mô tả bảng. Cách factor-head initialization cần thực nghiệm |
| Micro-KV | §3.2 p5: weighted prefix concatenation, optional | Compiler/composer có tensor interfaces; `training/factory.py` chủ động chặn bật KV | **Partial / Missing / Paper-unspecified**: chưa có attention-prefix injection, heads/positions/slots/init không được công bố. Không suy diễn cấu hình KV gốc |
| Router cosine/reranker | §3.2 p5; Table 6: K1=32, frozen BGE-base max192, fusion1, K≤8, temp1 | `router/model.py`, `router/reranker.py`: cosine top32, frozen BGE cross-encoder trên retrieval text, top-K | **Implemented / Paper-unspecified**: content hay retrieval_text làm cross-encoder input không được xác định đủ rõ |
| Router metadata | Table 6: 4 learned scalars; §3.2: confidence/answer-bearing/conflict/type | `router/model.py`: four weights **plus** learned semantic-type embedding; binary conflict membership | **Partial / Paper-unspecified**: encoding có thêm tham số, không thể gọi là đúng bốn scalar tổng cộng; abstraction_level không có bias riêng |
| Null-memory routing | §3.2/3.3: near-zero weights cho irrelevant nhưng top-K softmax tổng1; không nêu null candidate | `router/model.py`: learned null head; `training/loop.py`: thêm `-log(correct_decision)` vào route loss | **Deviation** có sẵn trước migration; thay normalization và objective. Cần quyết định nghiên cứu riêng, không xóa/chỉnh âm thầm |
| Composer / gate | Eq1 p5: weighted sum A và B; aggregated sigmoid gate | `composer/model.py`: đúng weighted factor sums, aggregate mask logits rồi sigmoid; gate nhân **một lần vào output delta** qua injector | **Implemented / Paper-unspecified**: factor composition khớp Eq1; gate placement một lần thay vì nhân hai factors là giả định ảnh hưởng g vs g² |
| Teacher/student | §3.2/3.3 pp5–6: cùng frozen base, teacher D+q, student q+memory | `training/system.py`, `training/loop.py`: teacher no_grad, student hook LoRA, forced-gold path, answer-position logits | **Implemented / Partial**: training prompt plain khác inference refusal/short-answer prompt; literals/chat template chưa công bố |
| LM + distillation | Eqs2–3: answer CE, KL teacher→student, τ1.5 | `training/loop.py`, `losses/objectives.py`: CE, KL/temperature, answer masks | **Implemented / Paper-unspecified**: CE mean vs paper notation sum; KL reduction/scaling và τ² không được chốt; không tự thêm τ² |
| Routing + rerank losses | Eq4; §3.3: weights0.5/1/1.5, listwise rerank CE | `losses/objectives.py`, `training/loop.py`: gold-only BCE trên stage1 similarities; listwise reranker+metadata candidates; thêm null term | **Partial / Deviation**: BCE không dùng final fused si như cách viết paper; per-element BCE weighting khác literal α chỉ đặt ở positive term trong Eq4. Reranker frozen nên metadata nhận phần gradient của rerank loss |
| Irrelevant/protection | Eqs5–6: factor norm, delta Frobenius; alternative null-space | Eq5 factor norms cho irrelevant; protection dùng actual `(α/r)gBA`; không có null-space projection | **Implemented / Paper-unspecified** cho L2; **Missing** alternative null-space. Null-space là lựa chọn thay thế, không chứng tỏ bắt buộc để chạy main variant |
| Sparse/composition/conflict/confidence | Eqs7–8 và đoạn robustness p6 | Per-valid-atom sigmoid density penalty; symmetric KL main/forced-gold; intra-group listwise CE; confidence MSE | **Implemented / Paper-unspecified**: forced weights theo stage1 scores, KL averaging, MSE/normalization và conflict edge cases là lựa chọn riêng |
| Curriculum | App D p15: 0/2000/6000/9000, K1/4/8/8; tăng irrelevant/sparse; max50 epochs | `training/curriculum.py`, `trainer.py`: stage boundaries/loss activation; configurable shortened schedules | **Implemented / Deviation** khi chạy diagnostic800 steps. Multipliers2, coefficients, seed, early-stop patience là giả định; thời điểm riêng cho rerank chưa hoàn toàn rõ |
| Optimization / precision | App C pp14–15; Table 6: AdamW, 1e-4/8e-5, wd.01, warmup500, cosine, clip1, bf16/fp32 | `configs/gemma_paper.yaml`, `qwen_paper.yaml`, `training/loop.py`, `factory.py`: khai báo/áp dụng phần lớn settings | **Implemented / Partial**: paper-like configs cố ý không runnable khi unknown weights=null; actual settings tùy reconstruction profile |
| Distributed/data loading | App C/Table6: 8×A10080GB NVLink, ZeRO2/DDP, workers8/prefetch4/persistent | `training/distributed.py` và integration workers có runtime/sampler/checkpoints; train CLI gọi local runtime, không DataLoader | **Partial / Missing / Deviation**: setting backend config không làm CLI thành DDP; `torchrun` CLI hiện nguy hiểm vì các process độc lập. Chưa xác minh GPU/NCCL/ZeRO2 |
| Data maxima / mixture | App C: context8192, atom96, answer128, atoms64, questions8; 5 train corpora, LongBench held-out | `_memory` truncate atom96; teacher tổng sequence cap; loop question cap; `data/prepare.py` chỉ download SQuAD, adapters cho corpora khác | **Partial / Missing**: max_atoms64 chưa enforce, answer không truncate độc lập128, teacher cap tính cả prompt/answer; mixture/splits/version đầy đủ chưa được tái tạo |
| Inference | §3.2/Fig3: offline compile, query-only routing+injection | `inference/build_memory.py`, `inference/query.py`, `answer_format.py`: bank, query, greedy generation, provenance/routing traces | **Implemented / Partial**: bank xây lại khi chạy evaluator, chưa có CLI kho persistent bank đầy đủ; query truncation/format train–eval không đồng nhất |
| Evaluation / refusal | Table1 F1/ROUGE-L; Table2 F1 against literal `unanswerable` on unrelated queries | `evaluation/cli.py`, `metrics.py`, `compare.py`: source join, multiref wordF1, ROUGE, routing recalls/MRR, latency, binary refusal F1 | **Implemented / Deviation**: `refusal_f1` là binary classification F1 trên mixed classes, không trực tiếp là Table2 target-string F1; tokenization/multilingual protocol chưa xác minh khớp paper |
| Baselines / efficiency / metrics | Tables1–3: D2Lckpt/raw/atom, Base+ctx, held-out LongBench; compile update latency/additional memory | `evaluation/baselines.py` có adapters và base references; chưa có training/checkpoints D2L; training/eval logging không phải protocol Table3 | **Partial / Missing**: chưa đo lại Table1–3, chưa có dedicated compile-only timing/baseline experiment; không dùng peak training VRAM thay additional compile memory |
| Checkpoints/resume | Paper không công bố format | `training/checkpoint.py`, `state.py`, `runtime/run.py`: committed saves, config checks, optimizer/RNG/scheduler/curriculum/sampler; tests equivalence/retention | **Implemented / Paper-unspecified**, với **Partial** dataset/model/prompt binding: fields tồn tại nhưng loop hiện chỉ populate/check config fingerprint, chưa pin content hashes của dataset/model/prompt |

## 4. Các thông số chưa công bố và hướng thực nghiệm

**Các giá trị sau là đề xuất để thử, chưa triển khai thành kiến trúc/loss mới và
không được ghi là giá trị gốc của paper.** Chỉ dùng validation để chọn; test và
LongBench held-out không được dùng để tune.

| Nhóm chưa biết | Giá trị hiện có / lựa chọn | Đề xuất thực nghiệm, không phải paper |
|---|---|---|
| Total-loss coefficients | Reconstruction: lm/distill/route/rerank=1; irrel=.1; protect=.001; sparse=.01; composition/conflict/confidence=.1 | Giữ lm=1 làm anchor; từng auxiliary weight ×{0.1,1,10}, ablation0, đo gradient norms/F1/refusal/retention, tránh grid đồng thời quá lớn |
| Sparse target ρ | .25 | {0.1,.25,.5}; ghi per-layer gate density, không chỉ scalar mean |
| Stage3/4 strength | irrel×2 và sparse×2 | {1,2,4}, giữ optimizer steps/dataset cố định |
| Probe count / prompt calibration | 2 probes/document, prompts/adapters reconstructed | {1,2,4}; human audit grounding, duplicate/leakage/answerability và canary thất bại trước chạy toàn bộ |
| Annotation decoding | M3.1 provider temperature1/top_p.95; effort max; budget8192 | So effort {medium,max}, budget {8192,16384,32768} theo cost/quota; dừng ở truncation thay vì chấp nhận XML thiếu. Không gọi temperature0 là deterministic |
| Metadata/type/conflict encoding | type scalar embedding, binary conflict membership | Cần duyệt thiết kế trước khi thử one-hot/embedding/conflict features; báo số trainable params và giữ baseline cũ |
| Gate placement / A-head init | gate-on-delta; normal A-head params | Thử g vs g² và output variance calibration chỉ trong experiment profile riêng; không sửa checkpoint kiến trúc hiện có |
| KL/listwise/confidence reductions | batchmean KL, normalized-gold CE, MSE confidence | Ghi reduction/token count/τ scaling; kiểm tra độ nhạy trên length-matched batches; paper không cho công thức confidence để coi MSE là official |
| Null decision | learned null candidate + null supervision | Cần quyết định riêng: giữ reconstruction hay thêm experiment paper-literal top-K; phải version kiến trúc/fingerprint và không load nhầm checkpoint |
| Early stopping / validation | Reconstruction patience5/min_delta0; notebook patience100 | {3,5,10}; cadence cố định, LM validation tách khỏi test; notebook800 chỉ diagnostic |
| Micro-KV | Disabled, shape/position/init unknown | Không đề xuất “giá trị paper”; cần nguồn bổ sung hoặc thiết kế nghiên cứu được duyệt. Tiếp tục fail-fast khi bật |
| Data/version/seed/mixture | SQuAD helper, seed fallback42; thiếu full split/mixture manifest | Pin corpus revisions, document-disjoint split hashes, seeds {7,42,123}; cân đối corpus theo plan công bố riêng; không trộn LongBench train |

Những giá trị **đã công bố** không được đánh dấu missing: rank8, α16, std.02,
dim256, encoder4, memory last4, targets q/v/o/down, BGE-base/max192, K1=32,
K=8, temperature router1, AdamW betas(.9,.999), wd.01, LR Gemma1e-4/Qwen8e-5,
warmup500/cosine/clip1, distillτ1.5, batch128, context8192/atom96/answer128,
atoms64/questions8, greedy≤128, epochs≤50, stage boundaries2000/6000/9000.

## 5. Migration/API contract và bảo toàn dữ liệu

“Offline atomization” là tiền xử lý trước training/inference, không có nghĩa HTTP
API có thể chạy khi ngắt Internet.

- Official endpoint cố định, HTTPS, header `Authorization: Bearer <MINIMAX_API_KEY>`
  và `Content-Type: application/json`; không follow redirect để tránh gửi key sang host khác.
- POST non-streaming, `model`, system/user `messages`, `max_completion_tokens`,
  `reasoning_split=true`; gửi `reasoning_effort` cho M3.1-Flash. Không gửi sampling
  fields bị provider bỏ qua; không giả định có thể disable thinking.
- Đọc `choices[0].message.content`, model/finish reason và `base_resp.status_code`.
  Sidecar lưu cả JSON body dạng text: reasoning_content, request ID, usage và errors
  còn nguyên trong đó; lưu thêm requested model, request parameters, latency,
  HTTP status, content SHA và raw-body SHA. Không lưu Authorization header.
- `<think>` chỉ loại khỏi **parser input**, không sửa `.content`/raw response đã lưu.
- HTTP401/402/403/429/5xx, redirect, JSON/schema lỗi, output rỗng, truncation,
  content_filter/tool_calls dừng pipeline, không tự lặp paid calls. 402 vẫn cần
  xử lý billing/quota; sửa code không tạo thêm credit.
- `--provider minimax` vẫn hợp lệ cho lệnh MiniMax cũ; bỏ backend GPT và mặc định
  lựa chọn provider từ môi trường. `--model`, `--max-completion-tokens`,
  `--reasoning-effort` là config rõ ràng; không đọc model/base URL từ biến cũ.
- `.env.example` không có khóa thật. File `.env` riêng của người dùng không được
  đọc/in giá trị hay sửa trong migration; hãy tự bỏ cấu hình cũ không dùng và
  **revoke API key đã lộ trong trao đổi trước**, tạo key mới trước mọi API call.
- Cache-only resume không đòi key và không gọi mạng; không reserialize cached rows
  khi không thay record. Sidecar cũ vẫn đọc được với optional fields mặc định null.
- `--retry-invalid` là thao tác thay record được yêu cầu rõ ràng; trước khi thay
  giữ snapshot toàn bộ bytes cũ, rồi atomic replace. Không xóa raw sidecar, không
  chuyển provenance lịch sử sang model mới. Để thí nghiệm thuần M3.1 dùng file mới,
  không resume vào tập cũ nhiều annotators.
- Schema-v1/source-answer migration, strict joins và semantics validation không
  được sửa. Các checkpoint/config cũ không được rewrite; redaction các tên token
  lịch sử vẫn giữ để không làm đổi fingerprint hay lộ credential cũ.

## 6. File sửa/xóa và giới hạn cleanup

| Thao tác | Files | Lý do |
|---|---|---|
| Sửa transport/CLI/provenance | `src/doc_to_atom/atomization/client.py`, `cli.py`, `provenance.py` | MiniMax-only, official HTTP contract, fail-fast, raw capture, cache/backup safety; bỏ OpenAIConfig/OpenAIAtomizer và retry đặc thù cũ |
| Sửa packaging | `pyproject.toml`, `uv.lock` | Thay direct dependency openai bằng httpx; lock bỏ openai, jiter, httpx2/httpcore2/httpx2-jsfetch/truststore không còn cần |
| Sửa notebooks/runtime | `notebook.py`, `notebook_molab_merged.py`, `src/doc_to_atom/runtime/molab_notebook.py` | Chỉ nhận MiniMax annotation key; không tự chọn GPT nếu có khóa cũ |
| Xóa | `notebook2.py` | Notebook CLI cũ trùng nhiệm vụ với merged notebook, hard-coded Molab path và GPT-only annotation; không có data/checkpoint/provenance trong file. Giữ hai variants còn lại vì native-kernel và subprocess-venv là hai cách chạy khác nhau |
| Sửa redaction comment | `src/doc_to_atom/credentials.py` | Giải thích legacy token names; **không đổi rules/hashes** |
| Sửa tests | `tests/test_atomization.py`, `test_atomization_join.py`, `test_atomization_provenance.py`, `test_molab_notebook_cells.py`, `test_molab_notebook_runtime.py` | Transport fakes dùng MiniMax; bỏ fallback import chết; giữ kiểm thử sidecar lịch sử |
| Thêm tests | `tests/test_minimax_client.py` | Request/schema/auth/finish/error tests, legacy cache + provenance, backup, không gọi mạng |
| Thêm template | `.env.example` | Credential placeholders, không chứa key thật |
| Sửa tài liệu hiện hành | `README.md`, `PAPER_SPEC.md`, `REPRODUCTION_NOTES.md`, `docs/IMPLEMENTATION_MAP.md`, `docs/LOCAL_MOLAB_RUNBOOK.md`, `docs/MEDIUM_LOCAL_RUNBOOK.md` | Commands hiện hành và sửa các khẳng định sai về null gate/supporting positives/completeness |
| Đánh dấu historical | `docs/PHASE_LOG.md`, `docs/ATOMIZATION_V2_ACCEPTANCE.md`, `docs/superpowers/plans/2026-10-17-atomization-datasets.md`, `2026-10-17-end-to-end-evaluation.md`, `2026-10-19-atomization-data-contract-v2.md`, `docs/superpowers/specs/2026-10-17-full-pipeline-readiness-design.md` | Không hướng dẫn chạy backend đã bỏ, nhưng giữ bằng chứng/phương án nghiên cứu lịch sử |
| Thêm báo cáo | `docs/MINIMAX_MIGRATION_PAPER_AUDIT.md` | Báo cáo này |

Các chuỗi tên provider cũ còn lại chỉ thuộc **provenance lịch sử, compatibility tests,
generic secret redaction hoặc tài liệu historical được đánh dấu**, không phải live
backend/dependency hay biến chọn GPT. Không đổi tên dataset lịch sử, không xóa
raw responses/research reports chỉ để grep không còn tên provider cũ. Scripts CLI
wrapper vẫn giữ vì README/runbooks và người dùng còn sử dụng; không coi wrapper
ngắn là code chết nếu nó là entry point công khai.

## 7. Verification local

| Kiểm chứng | Lệnh/phương pháp | Kết quả |
|---|---|---|
| Official API contract | Đọc hai tài liệu MiniMax trên; httpx stub responses | Endpoint/model/auth và request/response validated theo docs; chưa có live API canary |
| Dependency migration | `uv lock`; `uv sync --locked --extra train --extra test` | Thành công; SDK openai và dependencies chỉ phục vụ SDK đã được gỡ khỏi env/lock |
| Full unit/integration sau khi gỡ SDK | `.venv\Scripts\python.exe -m pytest tests -ra` | **291 passed, 1 skipped**, 39.49s; skip Molab2GPU bằng opt-in marker |
| Core invariants | Trong suite: source/schema-v2/migration/join/target tests; checkpoint snapshot/retention; local/distributed resume equivalence; eval generation/metrics/provenance | Đều pass; distributed regression là local CPU workers, không phải NCCL/DeepSpeed GPU certification |
| Notebook static graph | `uvx --from marimo==0.25.2 marimo check --strict notebook_molab_merged.py notebook.py` | Pass trên local, không chạy cell API/GPU |
| Notebook boot | `marimo export html --no-sandbox notebook_molab_merged.py -o <temp>.html` | Pass; trạng thái chưa bấm nút không gọi API/GPU |
| CLI/dependency/static checks | `doc2atom-atomize --help`, kiểm tra module inventory, Ruff F401/F821 trên transport/tests mới, `git diff --check` | Pass; `openai_installed=False`, `httpx=0.28.1`, CLI chỉ có provider minimax |
| Synthetic smoke | `python scripts/smoke_train.py --config configs/resource_constrained.yaml --output-dir <temp ngoài repo>` | Step1, finite loss ≈5.730617, checkpoint save/load; compiler/router tổng hợp, LM logits giả, không gọi base model hoặc API |
| Research artefacts | So SHA-256 mọi file `data/`, `outputs/` trước/sau | 51 files → 51 files, **0 changed entries**; baseline hash manifest lưu ngoài repo |

Smoke artefacts và pytest outputs nằm trong
`C:\Users\ASUS\AppData\Local\Temp\opencode`, không ghi đè experiment outputs.

## 8. Rủi ro và quyết định cần duyệt trước khi triển khai tiếp

1. **Research priority:** có giữ null-head/null-loss reconstruction hay thêm variant
   paper-literal? Không thể tháo null head khỏi checkpoint cũ rồi gọi là resume
   tương đương. Cần kiến trúc/version/ablation riêng.
2. **Reproducibility priority:** checkpoint format có dataset/model/prompt hash fields
   nhưng loop chưa populate/check đầy đủ. Regression hiện chứng minh resume với
   cùng input, không chứng minh từ chối dataset bị đổi. Muốn hoàn thiện binding cần
   migration/version policy cho checkpoint cũ, không tự thay fingerprint semantics.
3. **Data priority:** trước full run phải pin splits/revisions/mixture, enforce caps
   có chủ đích và đưa source-aware acceptance vào quy trình. CLI train hiện structural
   validation không tương đương source-aware gate của notebook/evaluator. Một số
   stats semantic errors không tự biến thành exception ở mọi entry point.
4. **Metrics priority:** cần xuất riêng target-string refusal F1 của Table2, không
   đổi định nghĩa field binary `refusal_f1` hiện tại làm sai các report cũ. Cần duyệt
   scoring protocol/version trước benchmark comparison.
5. **Runtime priority:** nối distributed CLI, dataloader và kiểm thử thực GPU là công
   việc riêng. Không launch nhiều process CLI hiện tại với cùng output directory.
6. **Annotation cost/quality:** tài liệu xác nhận model ID không bảo đảm account có
   quota. Budget8192 có thể truncation do thinking; XML của annotator mới cần canary
   được người dùng cho phép. Model sampling không hoàn toàn deterministic; lưu raw
   outputs để replay, không gọi lại rồi giả định cùng dữ liệu.
7. **Numerical reproduction:** chưa tái tạo đủ hai backbones, năm in-domain corpora,
   tám held-out subsets, D2L baselines và compile-efficiency protocol. Không đưa
   chỉ số paper vào report như kết quả đo của repository.

Các quyết định trên được báo cáo, **chưa triển khai thay đổi nghiên cứu** trong
migration này. Không sửa architecture/losses/metrics để che những sai khác.
