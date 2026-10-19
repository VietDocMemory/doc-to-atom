# /// script
# dependencies = [
#     "marimo>=0.25,<1",
#     "molab==0.7.12",
#     "python-dotenv>=1.0,<2",
# ]
# requires-python = ">=3.12"
# ///

import marimo

__generated_with = "0.25.1"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import os
    import shutil
    import subprocess
    import sys
    from datetime import datetime, timezone
    from pathlib import Path

    import marimo as mo

    return Path, datetime, mo, os, shutil, subprocess, sys, timezone


@app.cell
def _(Path, os, sys):
    NOTEBOOK_FILE = Path(__file__).resolve()
    NOTEBOOK_DIR = NOTEBOOK_FILE.parent

    # Expected Molab layout:
    # /marimo/.env, /marimo/notebook.py, /marimo/doc-to-atom/
    candidate_repo = NOTEBOOK_DIR / "doc-to-atom"
    if not (candidate_repo / "src" / "doc_to_atom").is_dir():
        if (NOTEBOOK_DIR / "src" / "doc_to_atom").is_dir():
            candidate_repo = NOTEBOOK_DIR
        else:
            raise FileNotFoundError(
                "Không tìm thấy repo doc-to-atom cạnh notebook.py. "
                "Hãy clone repo thành thư mục 'doc-to-atom'."
            )

    REPO_DIR = candidate_repo.resolve()
    SRC_DIR = REPO_DIR / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    os.chdir(REPO_DIR)

    from doc_to_atom.runtime.molab_notebook import (
        find_molab_env_file,
        find_offline_wandb_runs,
        load_molab_environment,
        resolve_latest_checkpoint,
    )

    ENV_FILE = find_molab_env_file(NOTEBOOK_FILE, REPO_DIR)
    SECRET_STATUS = load_molab_environment(ENV_FILE)

    return (
        ENV_FILE,
        NOTEBOOK_DIR,
        NOTEBOOK_FILE,
        REPO_DIR,
        SECRET_STATUS,
        find_offline_wandb_runs,
        resolve_latest_checkpoint,
    )


@app.cell
def _(ENV_FILE, REPO_DIR, SECRET_STATUS, mo):
    _secret_lines = "\n".join(
        f"- `{name}`: {'✅ available' if available else '❌ missing'}"
        for name, available in SECRET_STATUS.items()
    )
    mo.md(
        f"""
        # Doc-to-Atom trên Molab

        Notebook chỉ chạy tác vụ tốn API/GPU sau khi bạn bấm nút tương ứng.

        - Repo: `{REPO_DIR}`
        - Secrets file: `{ENV_FILE if ENV_FILE else 'không tìm thấy'}`

        {_secret_lines}

        Molab Secrets được đọc trực tiếp từ `.env` cạnh notebook với
        `override=False`; không sao chép secret vào repo và không ghi giá trị secret ra output.
        """
    )
    return


@app.cell
def _(Path, REPO_DIR, os, shutil, subprocess):
    DATA_DIR = REPO_DIR / "data"
    OUTPUT_DIR = REPO_DIR / "outputs" / "molab-single-gpu"
    RAW_DATA_PATH = DATA_DIR / "squad_molab_raw.jsonl"
    ATOM_DATA_PATH = DATA_DIR / "squad_molab_atoms.jsonl"
    FALLBACK_ATOM_DATA_PATH = DATA_DIR / "squad_small_atoms_openai.jsonl"
    PRED_EVAL_PATH = OUTPUT_DIR / "predictions.jsonl"
    CONFIG_PATH = REPO_DIR / "configs" / "reconstruction" / "molab_single_gpu.yaml"

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    home_bin = str(Path.home() / ".local" / "bin")
    UV_BIN = shutil.which("uv") or str(Path(home_bin) / "uv")

    def run_cmd(
        command: list[str], *, check: bool = True, extra_env: dict[str, str] | None = None
    ) -> subprocess.CompletedProcess:
        cmd = list(command)
        if cmd and cmd[0] == "uv":
            cmd[0] = UV_BIN
        current_env = os.environ.copy()
        current_env["PATH"] = f"{home_bin}:{current_env.get('PATH', '')}"
        src_path = str(REPO_DIR / "src")
        existing_pythonpath = current_env.get("PYTHONPATH")
        current_env["PYTHONPATH"] = (
            f"{src_path}:{existing_pythonpath}" if existing_pythonpath else src_path
        )
        if extra_env:
            current_env.update(extra_env)

        print(f"\n[RUNNING] {' '.join(cmd)}")
        return subprocess.run(cmd, cwd=REPO_DIR, env=current_env, check=check)

    def selected_atom_data() -> Path:
        if ATOM_DATA_PATH.is_file():
            return ATOM_DATA_PATH
        if FALLBACK_ATOM_DATA_PATH.is_file():
            return FALLBACK_ATOM_DATA_PATH
        raise FileNotFoundError(
            "Chưa có atomized dataset. Hãy chạy bước Chuẩn bị và atomize dữ liệu trước."
        )

    return (
        ATOM_DATA_PATH,
        CONFIG_PATH,
        FALLBACK_ATOM_DATA_PATH,
        OUTPUT_DIR,
        PRED_EVAL_PATH,
        RAW_DATA_PATH,
        UV_BIN,
        run_cmd,
        selected_atom_data,
    )


@app.cell
def _(UV_BIN, mo):
    setup_button = mo.ui.run_button(label="1. Cài dependencies và chạy unit tests")
    mo.vstack(
        [
            mo.md(
                f"""
                ## 1. Thiết lập môi trường

                `uv` được tìm thấy tại `{UV_BIN}`. Bước này chạy `uv sync` và test không gồm Molab multi-GPU gate.
                """
            ),
            setup_button,
        ]
    )
    return (setup_button,)


@app.cell
def _(run_cmd, setup_button):
    if setup_button.value:
        run_cmd(["uv", "sync", "--extra", "train", "--extra", "test"])
        run_cmd(["uv", "run", "pytest", "-q", "-m", "not molab"])
    return


@app.cell
def _(mo):
    environment_button = mo.ui.run_button(label="2. Kiểm tra GPU và credentials")
    mo.vstack([mo.md("## 2. Kiểm tra môi trường Molab"), environment_button])
    return (environment_button,)


@app.cell
def _(environment_button, run_cmd):
    if environment_button.value:
        _script = (
            "import os, torch; "
            "print('torch:', torch.__version__); "
            "print('cuda:', torch.cuda.is_available()); "
            "print('gpu_count:', torch.cuda.device_count()); "
            "print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'); "
            "print('HF_TOKEN:', 'available' if os.getenv('HF_TOKEN') else 'missing'); "
            "print('OPENAI_API_KEY:', 'available' if os.getenv('OPENAI_API_KEY') else 'missing'); "
            "print('WANDB_API_KEY:', 'available' if os.getenv('WANDB_API_KEY') else 'missing'); "
            "print('OPENAI_ANNOTATION_MODEL:', os.getenv('OPENAI_ANNOTATION_MODEL', 'default'))"
        )
        run_cmd(["uv", "run", "python", "-c", _script])
    return


@app.cell
def _(mo):
    prepare_button = mo.ui.run_button(label="3. Chuẩn bị, atomize và validate SQuAD nhỏ")
    mo.vstack(
        [
            mo.md(
                """
                ## 3. Chuẩn bị dữ liệu

                Nếu có `OPENAI_API_KEY`, bước này atomize bằng model trong
                `OPENAI_ANNOTATION_MODEL`. Nếu không, notebook chỉ dùng file atomized mẫu có sẵn.
                Atomization hỗ trợ resume và retry record có source span không hợp lệ.
                """
            ),
            prepare_button,
        ]
    )
    return (prepare_button,)


@app.cell
def _(
    ATOM_DATA_PATH,
    FALLBACK_ATOM_DATA_PATH,
    RAW_DATA_PATH,
    os,
    prepare_button,
    run_cmd,
):
    if prepare_button.value:
        if not RAW_DATA_PATH.is_file():
            run_cmd(
                [
                    "uv",
                    "run",
                    "scripts/prepare_dataset.py",
                    "--dataset",
                    "squad",
                    "--output",
                    str(RAW_DATA_PATH),
                    "--train-rows",
                    "16",
                    "--validation-rows",
                    "8",
                ]
            )

        if os.getenv("OPENAI_API_KEY"):
            run_cmd(
                [
                    "uv",
                    "run",
                    "doc2atom-atomize",
                    "--provider",
                    "openai",
                    "--input",
                    str(RAW_DATA_PATH),
                    "--output",
                    str(ATOM_DATA_PATH),
                    "--resume",
                    "--retry-invalid",
                ]
            )
            _data = ATOM_DATA_PATH
        elif FALLBACK_ATOM_DATA_PATH.is_file():
            print(
                "OPENAI_API_KEY không khả dụng; dùng dataset atomized mẫu "
                f"{FALLBACK_ATOM_DATA_PATH}."
            )
            _data = FALLBACK_ATOM_DATA_PATH
        else:
            raise RuntimeError("Thiếu OPENAI_API_KEY và không có atomized dataset mẫu.")

        run_cmd(
            [
                "uv",
                "run",
                "doc2atom-validate",
                "--input",
                str(_data),
                "--reject-span-mismatch",
            ]
        )
        run_cmd(["uv", "run", "scripts/audit_semantics.py", "--input", str(_data)])
    return


@app.cell
def _(mo):
    preflight_button = mo.ui.run_button(label="4. Chạy GPU preflight 1 optimizer step")
    mo.vstack(
        [
            mo.md(
                """
                ## 4. GPU preflight

                Chạy một optimizer step, một question và tắt BGE reranker. Mỗi lần bấm tạo output
                directory mới nên không đè run manifest cũ.
                """
            ),
            preflight_button,
        ]
    )
    return (preflight_button,)


@app.cell
def _(
    CONFIG_PATH,
    REPO_DIR,
    datetime,
    preflight_button,
    run_cmd,
    selected_atom_data,
    timezone,
):
    if preflight_button.value:
        _stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        _output = REPO_DIR / "outputs" / f"molab-gpu-check-{_stamp}"
        run_cmd(
            [
                "uv",
                "run",
                "doc2atom-train",
                "--config",
                str(CONFIG_PATH),
                "--data",
                str(selected_atom_data()),
                "--output",
                str(_output),
                "--max-steps",
                "1",
                "--max-questions",
                "1",
                "--no-reranker",
                "--wandb-mode",
                "offline",
            ]
        )
        print(f"GPU preflight hoàn tất: {_output}")
    return


@app.cell
def _(OUTPUT_DIR, mo):
    fresh_train_button = mo.ui.run_button(label="5A. Bắt đầu fresh training")
    resume_train_button = mo.ui.run_button(label="5B. Resume checkpoint mới nhất")
    mo.vstack(
        [
            mo.md(
                f"""
                ## 5. Single-GPU training

                Output cố định: `{OUTPUT_DIR}`.

                Fresh và resume là hai hành động riêng để tránh tự resume nhầm run.
                """
            ),
            mo.hstack([fresh_train_button, resume_train_button]),
        ]
    )
    return fresh_train_button, resume_train_button


@app.cell
def _(
    CONFIG_PATH,
    OUTPUT_DIR,
    fresh_train_button,
    resolve_latest_checkpoint,
    run_cmd,
    selected_atom_data,
):
    if fresh_train_button.value:
        if (OUTPUT_DIR / "manifests" / "run.json").exists() or resolve_latest_checkpoint(
            OUTPUT_DIR
        ):
            raise RuntimeError(
                f"{OUTPUT_DIR} đã chứa một run. Hãy dùng Resume hoặc đổi OUTPUT_DIR."
            )
        run_cmd(
            [
                "uv",
                "run",
                "doc2atom-train",
                "--config",
                str(CONFIG_PATH),
                "--data",
                str(selected_atom_data()),
                "--output",
                str(OUTPUT_DIR),
                "--wandb-mode",
                "online",
                "--wandb-project",
                "doc-to-atom",
            ]
        )
    return


@app.cell
def _(
    CONFIG_PATH,
    OUTPUT_DIR,
    resolve_latest_checkpoint,
    resume_train_button,
    run_cmd,
    selected_atom_data,
):
    if resume_train_button.value:
        _checkpoint = resolve_latest_checkpoint(OUTPUT_DIR)
        if _checkpoint is None:
            raise FileNotFoundError(
                f"Không tìm thấy {OUTPUT_DIR / 'checkpoints' / 'last.json'} để resume."
            )
        run_cmd(
            [
                "uv",
                "run",
                "doc2atom-train",
                "--config",
                str(CONFIG_PATH),
                "--data",
                str(selected_atom_data()),
                "--output",
                str(OUTPUT_DIR),
                "--resume",
                "latest",
                "--wandb-mode",
                "online",
                "--wandb-project",
                "doc-to-atom",
            ]
        )
    return


@app.cell
def _(mo):
    sync_button = mo.ui.run_button(label="6. Đồng bộ các W&B offline run")
    score_button = mo.ui.run_button(label="7. Chấm predictions JSONL")
    mo.vstack(
        [
            mo.md(
                """
                ## 6–7. Đồng bộ và evaluation

                Evaluation CLI hiện chỉ chấm file predictions đã tồn tại; notebook không truyền raw
                dataset vào held-out evaluator vì đó là sai schema.
                """
            ),
            mo.hstack([sync_button, score_button]),
        ]
    )
    return score_button, sync_button


@app.cell
def _(OUTPUT_DIR, find_offline_wandb_runs, run_cmd, sync_button):
    if sync_button.value:
        _runs = find_offline_wandb_runs(OUTPUT_DIR / "wandb")
        if not _runs:
            print("Không tìm thấy W&B offline run để đồng bộ.")
        for _run in _runs:
            run_cmd(["uv", "run", "wandb", "sync", str(_run)], check=False)
    return


@app.cell
def _(PRED_EVAL_PATH, run_cmd, score_button):
    if score_button.value:
        if not PRED_EVAL_PATH.is_file():
            raise FileNotFoundError(
                f"Chưa có {PRED_EVAL_PATH}. Evaluation CLI hiện không tự generate predictions."
            )
        run_cmd(
            [
                "uv",
                "run",
                "doc2atom-evaluate",
                "--predictions",
                str(PRED_EVAL_PATH),
            ]
        )
    return


if __name__ == "__main__":
    app.run()
