# /// script
# dependencies = [
#     "marimo==0.25.2",
#     "molab==0.7.12",
# ]
# requires-python = ">=3.13"
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    import hashlib
    import json
    import os
    import shutil
    from pathlib import Path

    import marimo as mo

    return Path, hashlib, json, mo, os, shutil


@app.cell
def _(Path, shutil):
    NOTEBOOK_DIR = Path(__file__).resolve().parent
    # Molab normally opens /marimo/notebook.py beside /marimo/doc-to-atom.
    # Also support opening the notebook from inside the repository.
    REPO_DIR = next(
        (_candidate for _root in (NOTEBOOK_DIR, Path.cwd())
         for _candidate in (_root, _root / "doc-to-atom")
         if (_candidate / "src" / "doc_to_atom").is_dir()),
        NOTEBOOK_DIR / "doc-to-atom",
    ).resolve()
    REPO_URL = "https://github.com/VietDocMemory/doc-to-atom.git"
    BRANCH = "main"
    UV = shutil.which("uv") or str(Path.home() / ".local" / "bin" / "uv")
    return BRANCH, NOTEBOOK_DIR, REPO_DIR, REPO_URL, UV


@app.cell
def _(NOTEBOOK_DIR, Path, REPO_DIR, UV, hashlib, os):
    def secret_env_file():
        for candidate in (NOTEBOOK_DIR / ".env", REPO_DIR.parent / ".env", REPO_DIR / ".env"):
            if candidate.is_file():
                return candidate
        return None

    def run_cmd(args, *, cwd=None, capture=False):
        import subprocess # Local import to break the DAG cycle
        import textwrap
        cmd = list(map(str, args))
        # Embedded scripts retain notebook indentation; python -c must not.
        if "-c" in cmd:
            script_index = cmd.index("-c") + 1
            cmd[script_index] = textwrap.dedent(cmd[script_index])
        if cmd[0] == "uv":
            if not Path(UV).is_file():
                raise FileNotFoundError("uv is not available in this Molab session")
            cmd[0] = UV
            if len(cmd) > 1 and cmd[1] == "run" and secret_env_file() is not None:
                # uv reads Molab's /marimo/.env without copying it into the repo.
                cmd[2:2] = ["--env-file", str(secret_env_file())]
        env = os.environ.copy()
        env["PATH"] = str(Path.home() / ".local" / "bin") + os.pathsep + env.get("PATH", "")
        env.pop("VIRTUAL_ENV", None)  # avoid uv overriding its own .venv selection
        # The notebook sandbox may select its own project/Python via these vars.
        for key in ("UV_PROJECT_ENVIRONMENT", "UV_PROJECT", "UV_WORKING_DIRECTORY", "UV_PYTHON"):
            env.pop(key, None)
        # Do not print environment variables or secrets.
        print("[RUNNING]", " ".join(cmd))
        completed = subprocess.run(cmd, cwd=cwd or REPO_DIR, env=env, check=True,
                                   text=True, capture_output=capture)
        return completed

    def sha256_file(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    return run_cmd, sha256_file


@app.cell
def _(mo):
    setup_button = mo.ui.run_button(label="1. Git pull/clone + uv sync")
    dataset_choice = mo.ui.dropdown(
        options={"Existing SQuAD (squad_molab)": "squad_molab",
                 "Medium SQuAD": "medium_squad",
                 "In-domain train/validation": "in_domain_train_val"},
        value="Existing SQuAD (squad_molab)", label="Dataset")
    mode_choice = mo.ui.dropdown(
        options={"Smoke (1 optimizer step)": "smoke",
                 "Standard single-GPU config": "standard",
                 "4-stage diagnostic (800 steps)": "four_stage"},
        value="4-stage diagnostic (800 steps)", label="Training profile")
    tracking_choice = mo.ui.dropdown(
        options={"Offline": "offline", "Online (W&B)": "online", "Disabled": "disabled"},
        value="Offline", label="Weights & Biases")
    run_tag = mo.ui.text(value="exp01", label="Run ID (letters, numbers, _ or -)")
    prepare_button = mo.ui.run_button(label="2a. Prepare/Resume atomization (API may cost)")
    validate_button = mo.ui.run_button(label="2b. Validate existing raw + atoms")
    config_button = mo.ui.run_button(label="3. Save/verify experiment config")
    preflight_button = mo.ui.run_button(label="4. GPU preflight (1 step)")
    train_button = mo.ui.run_button(label="5. Train (new run)")
    resume_button = mo.ui.run_button(label="Resume exact checkpoint")
    evaluate_button = mo.ui.run_button(label="6. Evaluate best checkpoint")
    report_button = mo.ui.run_button(label="7. View report")
    mo.vstack([
        mo.md("# Doc-to-Atom · Molab unified notebook\n"
              "All expensive API/GPU steps require an explicit click. "
              "The CLI runs in the repository's uv environment."),
        setup_button,
        mo.hstack([dataset_choice, mode_choice, tracking_choice, run_tag]),
        mo.hstack([prepare_button, validate_button]),
        mo.hstack([config_button, preflight_button]),
        mo.hstack([train_button, resume_button]),
        mo.hstack([evaluate_button, report_button]),
        mo.md("**4-stage profile:** 200/400/600 boundaries, 800 steps total, "
              "Top-K 1 → 4 → 8 → 8. This is a shortened diagnostic, NOT the paper schedule. "
              "With `stage4_only_best`, `best.json` appears only after stage-4 validation. "
              "Keep the same profile, run ID, dataset and tracking mode when resuming."),
    ])
    return (
        config_button,
        dataset_choice,
        evaluate_button,
        mode_choice,
        preflight_button,
        prepare_button,
        report_button,
        resume_button,
        run_tag,
        setup_button,
        tracking_choice,
        train_button,
        validate_button,
    )


@app.cell
def _(BRANCH, REPO_DIR, REPO_URL, run_cmd, setup_button):
    if setup_button.value:
        if (REPO_DIR / ".git").exists():
            _status = run_cmd(["git", "status", "--porcelain"], capture=True).stdout
            if _status.strip():
                # Uploaded datasets/notebooks must not block environment setup.
                print("Local changes detected; skipping git pull and keeping all files.")
            else:
                run_cmd(["git", "pull", "--ff-only", "origin", BRANCH])
        elif REPO_DIR.exists():
            raise FileExistsError(f"Existing directory is not a git repo: {REPO_DIR}")
        else:
            REPO_DIR.parent.mkdir(parents=True, exist_ok=True)
            run_cmd(["git", "clone", "--branch", BRANCH, REPO_URL, str(REPO_DIR)],
                    cwd=REPO_DIR.parent)
        # Use the project's Python, not the Molab kernel's Python/dependencies.
        # Single-GPU training does not need the train extra (DeepSpeed build).
        run_cmd(["uv", "sync", "--python", "3.12", "--extra", "test"])
        run_cmd(["uv", "run", "python", "-c",
                 "import doc_to_atom, torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"])
    return


@app.cell
def _(REPO_DIR, dataset_choice, mo, mode_choice, run_tag, tracking_choice):
    import re as _re

    if not _re.fullmatch(r"[A-Za-z0-9_-]+", run_tag.value):
        raise ValueError("Run ID must contain only letters, numbers, _ or -")
    prefix = dataset_choice.value
    raw_path = REPO_DIR / "data" / f"{prefix}_raw.jsonl"
    atoms_path = REPO_DIR / "data" / f"{prefix}_atoms.jsonl"
    # Separate output roots protect old runs and prevent config/dataset collisions.
    output_dir = (REPO_DIR / "outputs" / "molab-unified" / prefix /
                  mode_choice.value / run_tag.value)
    notebook_config = output_dir / "configs" / "notebook.yaml"
    # Keep preflight outside the run-ID namespace (exp01-preflight is a valid ID).
    preflight_dir = REPO_DIR / "outputs" / "molab-preflight" / prefix / mode_choice.value / run_tag.value
    mo.md(f"**Paths** · raw: `{raw_path}` · atoms: `{atoms_path}` · "
          f"run: `{output_dir}` · W&B: `{tracking_choice.value}`")
    return atoms_path, notebook_config, output_dir, preflight_dir, raw_path


@app.cell
def _(REPO_DIR, atoms_path, raw_path, run_cmd):
    def validate_pair():
        if not raw_path.is_file() or not atoms_path.is_file():
            raise FileNotFoundError(f"Missing raw/atomized source pair: {raw_path}, {atoms_path}")
        # Strong source-aware validation, NOT just training's standalone structural check.
        check = r'''
    import json,sys
    from doc_to_atom.data.io import validate_jsonl
    from doc_to_atom.data.semantic_validation import load_source_records
    raw,atoms=sys.argv[1:3]
    stats=validate_jsonl(atoms,source_path=raw,reject_span_mismatch=True)
    print(json.dumps(stats,ensure_ascii=False,indent=2))
    if stats.get("errors",0) or stats.get("invalid_spans",0) or stats.get("source_answer_mismatches",0) or stats.get("parse_errors",0):
        raise SystemExit("Source contract validation failed")
    if stats.get("records") != len(load_source_records(raw)):
        raise SystemExit("Some source records are missing from atomized data")
    '''
        run_cmd(["uv", "run", "python", "-c", check,
                 str(raw_path), str(atoms_path)])
        return True

    def ensure_repo_ready():
        if not (REPO_DIR / ".venv").is_dir():
            raise RuntimeError("Run step 1 (uv sync) before using train/evaluate")

    return ensure_repo_ready, validate_pair


@app.cell
def _(
    atoms_path,
    ensure_repo_ready,
    prepare_button,
    raw_path,
    run_cmd,
    validate_button,
    validate_pair,
):
    if prepare_button.value:
        import subprocess
        ensure_repo_ready()
        if not raw_path.exists() and raw_path.name == "squad_molab_raw.jsonl":
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            run_cmd(["uv", "run", "scripts/prepare_dataset.py", "--dataset", "squad",
                     "--output", str(raw_path), "--train-rows", "16",
                     "--validation-rows", "8"])
        if not raw_path.exists():
            raise FileNotFoundError("Upload matching raw JSONL to data/ first; do not use atoms as raw")
        complete = False
        if atoms_path.is_file():
            try:
                complete = validate_pair()
            except subprocess.CalledProcessError:
                print("Cached atomization incomplete or invalid; retrying with --resume --retry-invalid")
        if not complete:
            provider_code = ("import os;print('openai' if os.getenv('OPENAI_API_KEY') "
                             "else ('minimax' if os.getenv('MINIMAX_API_KEY') "
                             "and os.getenv('MINIMAX_BASE_URL') else 'none'))")
            provider = run_cmd(["uv", "run", "python", "-c", provider_code],
                               capture=True).stdout.strip()
            if provider not in ("openai", "minimax"):
                raise RuntimeError("Atomization requires OPENAI_API_KEY or MiniMax credentials")
            run_cmd(["uv", "run", "doc2atom-atomize", "--provider", provider,
                     "--input", str(raw_path), "--output", str(atoms_path),
                     "--resume", "--retry-invalid"])
        else:
            print("Cached atomization fully validated; no paid API call needed")
        validate_pair()
    if validate_button.value:
        ensure_repo_ready()
        validate_pair()
    return


@app.cell
def _(
    REPO_DIR, config_button, ensure_repo_ready, mode_choice,
    notebook_config, run_cmd, tracking_choice,
):
    def prepare_config():
        ensure_repo_ready()
        if mode_choice.value == "four_stage":
            base = REPO_DIR / "configs/reconstruction/molab_blackwell_1gpu.yaml"
        else:
            base = REPO_DIR / "configs/reconstruction/molab_single_gpu.yaml"
        # All config mutation happens in the project venv, not notebook kernel.
        builder = r'''
    import sys,yaml
    from pathlib import Path
    from doc_to_atom.config import load_and_resolve_config,validate_config,config_fingerprint
    base,destination,mode,tracking=sys.argv[1:5]
    config=load_and_resolve_config(base,require_runnable=True)
    config["runtime"]["wandb_mode"]=tracking
    config["runtime"]["checkpoint_upload"]="best_weights" if tracking=="online" else "none"
    if mode=="smoke":
        config["profile"]={"name":"molab-unified-smoke","family":"experiments"}
        config["training"]["max_steps"]=1
        config["training"]["max_epochs"]=10
        config["training"]["gradient_accumulation_steps"]=1
        config["training"]["effective_global_batch"]=1
        config["training"]["micro_batch_size"]=1
        config["training"]["validation_every_steps"]=1
        config["training"]["checkpoint_every_steps"]=1
        config["training"]["stage4_only_best"]=False
        config["data"]["max_questions_per_sample"]=1
    elif mode=="four_stage":
        config["profile"]={"name":"molab-unified-four-stage","family":"experiments"}
        config["deviations"]=[*config.get("deviations",[]),{
          "id":"MOLAB-UNIFIED-SHORTENED-STAGES",
          "description":"800 optimizer steps, 200 steps each curriculum stage",
          "rationale":"One-GPU diagnostic; not the paper training schedule",
          "affected_metrics":["answer_f1","refusal_f1","training_loss"]}]
        config["training"].update({"micro_batch_size":1,"gradient_accumulation_steps":16,
          "effective_global_batch":16,"curriculum_boundaries":[200,400,600],
          "initial_curriculum_step":0,"max_steps":800,"max_epochs":10000,
          "validation_every_steps":200,"checkpoint_every_steps":200,
          "stage4_only_best":True,"early_stopping_patience":100})
        config["data"].update({"max_context_tokens":1024,"max_atom_tokens":96,
          "max_answer_tokens":64,"max_questions_per_sample":3})
        config["runtime"].update({"distributed_backend":"none",
          "reranker_device":"cpu","gradient_checkpointing":False})
    validate_config(config,require_runnable=True)
    path=Path(destination)
    if path.exists():
        old=load_and_resolve_config(path,require_runnable=True)
        if config_fingerprint(old)!=config_fingerprint(config):
            raise SystemExit("Existing run config differs. Choose a NEW run ID; never overwrite.")
        print("Reusing identical config",path,config_fingerprint(old))
    else:
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding="utf-8")
        restored=load_and_resolve_config(path,require_runnable=True)
        assert config_fingerprint(restored)==config_fingerprint(config)
        print("Wrote config and verified round-trip fingerprint",config_fingerprint(config))
    '''
        run_cmd(["uv", "run", "python", "-c", builder,
                 str(base), str(notebook_config), mode_choice.value, tracking_choice.value])
        return notebook_config
    if config_button.value:
        prepare_config()
    return (prepare_config,)


@app.cell
def _(
    atoms_path,
    ensure_repo_ready,
    json,
    notebook_config,
    preflight_button,
    preflight_dir,
    prepare_config,
    raw_path,
    run_cmd,
    sha256_file,
    validate_pair,
):
    def preflight_marker():
        return preflight_dir / "verified.json"

    def expected_preflight_meta():
        commit=run_cmd(["git","rev-parse","HEAD"],capture=True).stdout.strip()
        return {"git_commit":commit,"config_sha256":sha256_file(notebook_config),
                "raw_sha256":sha256_file(raw_path),
                "atoms_sha256":sha256_file(atoms_path)}

    if preflight_button.value:
        ensure_repo_ready()
        validate_pair()
        prepare_config()
        meta=expected_preflight_meta()
        if preflight_marker().is_file():
            saved=json.loads(preflight_marker().read_text(encoding="utf-8"))
            if saved != meta:
                raise ValueError("Preflight belongs to another config/dataset: use a new Run ID")
            print("Verified preflight already exists. No extra GPU run.")
        else:
            if (preflight_dir / "manifests/run.json").exists():
                raise FileExistsError("Preflight run already exists; review it before retrying")
            run_cmd(["uv", "run", "python", "-c",
                     "import torch; assert torch.cuda.is_available(), 'Enable a GPU in Molab before preflight'"])
            run_cmd(["uv","run","doc2atom-train","--config",str(notebook_config),
                     "--data",str(atoms_path),"--output",str(preflight_dir),
                     "--max-steps","1","--max-questions","1",
                     "--wandb-mode","disabled"])
            _result_file=preflight_dir / "reports/training_result.json"
            _result=json.loads(_result_file.read_text(encoding="utf-8"))
            assert _result["training"]["optimizer_steps"]==1, "GPU preflight did not complete 1 step"
            preflight_marker().write_text(json.dumps(meta,indent=2)+"\n",encoding="utf-8")
            print("GPU preflight succeeded; original run has not started.")
    return expected_preflight_meta, preflight_marker


@app.cell
def _(
    atoms_path,
    ensure_repo_ready,
    expected_preflight_meta,
    json,
    notebook_config,
    output_dir,
    preflight_marker,
    prepare_config,
    resume_button,
    run_cmd,
    tracking_choice,
    train_button,
    validate_pair,
):
    if train_button.value or resume_button.value:
        if train_button.value and resume_button.value:
            raise ValueError("Train and Resume cannot be launched together")
        ensure_repo_ready()
        validate_pair()
        prepare_config()
        cmd=["uv","run","doc2atom-train","--config",str(notebook_config),
             "--data",str(atoms_path),"--output",str(output_dir),
             "--wandb-mode",tracking_choice.value,"--wandb-project","doc-to-atom"]
        if resume_button.value:
            if not (output_dir / "checkpoints/last.json").is_file():
                raise FileNotFoundError("Cannot resume: missing exact-state checkpoints/last.json")
            cmd += ["--resume","latest"]
        else:
            if (output_dir / "manifests/run.json").exists():
                raise FileExistsError("Run already started. Use Resume or a new Run ID")
            if not preflight_marker().is_file():
                raise RuntimeError("Run GPU preflight first")
            if json.loads(preflight_marker().read_text(encoding="utf-8")) != expected_preflight_meta():
                raise RuntimeError("Preflight fingerprint differs; rerun with a new Run ID")
        run_cmd(cmd)
    return


@app.cell
def _(
    atoms_path,
    ensure_repo_ready,
    evaluate_button,
    json,
    mo,
    output_dir,
    raw_path,
    run_cmd,
    sha256_file,
    validate_pair,
):
    if evaluate_button.value:
        ensure_repo_ready()
        validate_pair()
        snapshot=output_dir / "config/resolved.yaml"
        best=output_dir / "checkpoints/best.json"
        if not best.is_file():
            raise FileNotFoundError("best.json missing: stage-4 may not have been reached; resume training")
        if not snapshot.is_file():
            raise FileNotFoundError("Missing training config/resolved.yaml")
        pointer=json.loads(best.read_text(encoding="utf-8"))
        step=pointer["optimizer_step"]
        predictions=output_dir / "predictions" / f"validation-best-step{step}.jsonl"
        if predictions.is_file():
            # Existing predictions are never blindly accepted. Source-aware scoring
            # refuses stale gold fields and joins on question_id.
            _first_line=next((x for x in predictions.read_text(encoding="utf-8").splitlines() if x.strip()), None)
            if _first_line is None:
                raise ValueError(f"Predictions file is empty; review it before retrying: {predictions}")
            first=json.loads(_first_line)
            if (first.get("source_fingerprint") != sha256_file(raw_path) or
                first.get("atomized_fingerprint") != sha256_file(atoms_path) or
                first.get("checkpoint_step") != step):
                raise ValueError("Existing predictions belong to different checkpoint/dataset")
            _scored=run_cmd(["uv","run","doc2atom-evaluate","--predictions",str(predictions),
                            "--data",str(atoms_path),"--source",str(raw_path)], capture=True)
            # The score-only CLI returns JSON but does not save the metrics file.
            _metrics_path=output_dir / "metrics/evaluation_validation.json"
            _metrics_path.parent.mkdir(parents=True, exist_ok=True)
            _metrics_path.write_text(json.dumps(json.loads(_scored.stdout), ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
        else:
            run_cmd(["uv","run","doc2atom-evaluate","--config",str(snapshot),
                     "--data",str(atoms_path),"--source",str(raw_path),
                     "--checkpoint",str(best),"--output",str(predictions),
                     "--split","validation","--wandb-mode","disabled"])
        mo.output.append(mo.md(f"Validation predictions: `{predictions}`"))
    return


@app.cell
def _(json, mo, output_dir, report_button, run_cmd, sha256_file):
    if report_button.value:
        metrics=output_dir / "metrics/evaluation_validation.json"
        training=output_dir / "reports/training_result.json"
        if training.is_file():
            result=json.loads(training.read_text(encoding="utf-8"))
            brief=result.get("training",{})
            mo.output.append(mo.md(
                f"## Training · optimizer steps: **{brief.get('optimizer_steps')}**, "
                f"early stopped: **{brief.get('early_stopped')}**, "
                f"best validation LM: **{brief.get('best_validation_lm')}**"))
        if metrics.is_file():
            measured=json.loads(metrics.read_text(encoding="utf-8"))
            rows=[{"Metric":k,"Value":measured.get(k)} for k in
                  ("count","f1","exact_match","rouge_l","refusal_f1",
                   "false_refusal_rate","false_answer_rate","recall_at_1",
                   "recall_at_4","recall_at_8","mrr","latency_ms",
                   "checkpoint_step")]
            mo.output.append(mo.md("## Evaluation"))
            mo.output.append(mo.ui.table(rows))
        if (output_dir / "train_log.jsonl").is_file() and metrics.is_file():
            # Report writes a unique path and never overwrites old artifacts.
            _report_key=sha256_file(output_dir / "train_log.jsonl")[:12] + "-" + sha256_file(metrics)[:12]
            report_path=output_dir / "reports" / f"medium_summary_{_report_key}.json"
            if not report_path.exists():
                run_cmd(["uv","run","python","scripts/report_medium_run.py",
                         "--run",str(output_dir),"--output",str(report_path)])
            summary=json.loads(report_path.read_text(encoding="utf-8"))
            stages=[{"Stage":stage,"Updates":v["optimizer_updates"],
                     "Top-K":str(v["top_k"]),"LM loss":v["loss_means"].get("lm"),
                     "Irrelevant loss":v["loss_means"].get("irrelevant")}
                    for stage,v in summary["stages"].items()]
            mo.output.append(mo.md("## Observed stage losses (not paper values)"))
            mo.output.append(mo.ui.table(stages))
        if not training.is_file() and not metrics.is_file():
            mo.output.append(mo.md("No training or evaluation results in this run yet."))
    return


if __name__ == "__main__":
    app.run()
