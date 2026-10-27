# /// script
# dependencies = ["marimo>=0.25,<1", "pandas>=2,<3"]
# requires-python = ">=3.12"
# ///
"""Molab single-GPU notebook. Install this checkout and dependencies in the kernel first.

No cell clones a repository, installs packages, or starts a shell. The notebook
must run with the same Python environment as doc-to-atom (including torch).
All API- and GPU-intensive work requires an explicit button click.
"""

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="full")


@app.cell
def _():
    import argparse
    import gc
    import importlib.util
    import json
    import os
    import sys
    from pathlib import Path

    import marimo as mo
    import torch

    notebook_path = Path(__file__).resolve()
    notebook_dir = notebook_path.parent
    repo_dir = next(
        (candidate for candidate in (notebook_dir / "doc-to-atom", notebook_dir)
         if (candidate / "src" / "doc_to_atom").is_dir()),
        None,
    )
    if repo_dir is None:
        raise FileNotFoundError("Place the doc-to-atom checkout beside notebook.py or inside its directory")
    repo_dir = repo_dir.resolve()
    for source_dir in (repo_dir / "src", repo_dir):
        if str(source_dir) not in sys.path:
            sys.path.insert(0, str(source_dir))
    os.chdir(repo_dir)  # HF model/config paths in this project are repository-relative.

    from doc_to_atom.runtime.molab_notebook import find_molab_env_file, load_molab_environment

    env_path = find_molab_env_file(notebook_path, repo_dir)
    secret_status = load_molab_environment(env_path)  # never display secret values
    os.environ["WANDB_MODE"] = "offline"
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    gpu_count = torch.cuda.device_count()
    gpu_name = torch.cuda.get_device_name(0) if gpu_count else None
    gpu_gib = torch.cuda.get_device_properties(0).total_memory / 2**30 if gpu_count else None
    bf16 = torch.cuda.is_bf16_supported() if gpu_count else False
    flash_available = importlib.util.find_spec("flash_attn") is not None
    hardware_ok = (gpu_count == 1 and gpu_gib is not None and gpu_gib >= 80 and bf16
                   and gpu_name is not None and "RTX PRO 6000" in gpu_name.upper()
                   and "BLACKWELL" in gpu_name.upper())
    prepare_button = mo.ui.run_button(label="1. Resume atomization + validate (API costs)")
    dataset_choice = mo.ui.dropdown(options={"Medium SQuAD": "medium_squad",
                                             "In-domain train/validation": "in_domain_train_val"},
                                    value="Medium SQuAD", label="Dataset")
    train_button = mo.ui.run_button(label="2. Preflight + train four stages")
    resume_button = mo.ui.run_button(label="Resume exact run from last.json")
    evaluate_button = mo.ui.run_button(label="3. Evaluate stage-4 best (validation)")
    mo.output.append(mo.md(
        f"# Doc-to-Atom / Molab (native Python)\n"
        f"GPU: {gpu_count} × {gpu_name or 'none'}; VRAM: {gpu_gib or 0:.1f} GiB; "
        f"BF16: {bf16}; flash-attn installed: {flash_available}. "
        f"{'Ready for single-GPU preflight.' if hardware_ok else 'GPU preflight blocked.'}\n\n"
        f"Secrets: OpenAI={secret_status['OPENAI_API_KEY']}, "
        f"MiniMax={bool(os.getenv('MINIMAX_API_KEY'))}, "
        f"HF_TOKEN={secret_status['HF_TOKEN']}. W&B: offline. "
        f"No DataLoader is constructed by the training loop: num_workers=2, "
        f"persistent_workers and prefetch_factor are **not implemented**. "
        f"CPU Torch threads: {torch.get_num_threads()}."
    ))
    mo.output.append(mo.hstack([dataset_choice, prepare_button, train_button,
                                resume_button, evaluate_button]))
    return (argparse, dataset_choice, evaluate_button, gc, hardware_ok, json, mo,
            os, prepare_button, repo_dir, resume_button, torch, train_button)


@app.cell
def _(dataset_choice, mo, os, prepare_button, repo_dir):
    from doc_to_atom.atomization.cli import atomize_file
    from doc_to_atom.data.io import validate_jsonl
    from doc_to_atom.data.semantic_validation import load_source_records

    raw_path = repo_dir / "data" / f"{dataset_choice.value}_raw.jsonl"
    atoms_path = repo_dir / "data" / f"{dataset_choice.value}_atoms.jsonl"
    mo.output.append(mo.md(
        f"## Data\nSource: `{raw_path}`; atoms: `{atoms_path}`. "
        "Upload/choose matching raw and atomized files before running; "
        "an atomized JSONL cannot be used as raw source. The OpenAI/MiniMax "
        "provider is selected from available credentials."
    ))
    if prepare_button.value:
        if not raw_path.is_file():
            raise FileNotFoundError(f"Upload the raw source file first: {raw_path}")
        source_count = len(load_source_records(raw_path))
        existing_ok = False
        if atoms_path.is_file():
            try:
                previous = validate_jsonl(atoms_path, source_path=raw_path, reject_span_mismatch=True)
                existing_ok = previous["errors"] == 0 and previous["records"] == source_count
            except ValueError:
                pass  # Invalid cached spans need explicit review; --resume won't silently replace them.
        if not existing_ok:
            if not os.getenv("OPENAI_API_KEY") and not (os.getenv("MINIMAX_API_KEY") and os.getenv("MINIMAX_BASE_URL")):
                raise RuntimeError("Incomplete atoms: configure a provider key before resuming")
            provider = "openai" if os.getenv("OPENAI_API_KEY") else "minimax"
            with mo.status.spinner(title="Resuming atomization (no cached records overwritten)"):
                atomize_file(raw_path, atoms_path, provider=provider, resume=True)
        _stats = validate_jsonl(atoms_path, source_path=raw_path, reject_span_mismatch=True)
        if _stats["errors"] or _stats["invalid_spans"] or _stats["records"] != source_count:
            raise ValueError(f"Source contract failed; do not train: {_stats}")
        mo.output.append(mo.md(f"Validated {_stats['records']} records / {_stats['atoms']} atoms. "
                               "Invalid records are NOT silently discarded; repair/retry at source first."))
    return atoms_path, raw_path, validate_jsonl


@app.cell
def _(atoms_path, mo, repo_dir):
    import yaml

    from doc_to_atom.config import load_and_resolve_config, validate_config
    from doc_to_atom.training.curriculum import stage_for_step

    base_path = repo_dir / "configs/reconstruction/molab_blackwell_1gpu.yaml"
    output_dir = repo_dir / "outputs/molab/gemma_full"
    notebook_config = output_dir / "configs/notebook.yaml"
    # This is a continuous, deliberately shortened curriculum (not the paper schedule).
    stage_steps = 200
    if notebook_config.exists():
        config = load_and_resolve_config(notebook_config, require_runnable=True)
    else:
        config = load_and_resolve_config(base_path, require_runnable=True)
        config["profile"] = {"name": "molab-notebook-four-stage", "family": "experiments"}
        config["deviations"] = [*config.get("deviations", []), {
            "id": "MOLAB-NOTEBOOK-SHORTENED-STAGES",
            "description": "One continuous 800-step run with 200 steps per curriculum stage",
            "rationale": "Four-stage diagnostic on one Blackwell GPU, not the paper's schedule",
            "affected_metrics": ["answer_f1", "refusal_f1", "training_loss"],
        }]
        config["training"].update({
            "micro_batch_size": 1, "gradient_accumulation_steps": 16,
            "effective_global_batch": 16,
            "curriculum_boundaries": [200, 400, 600],
            "initial_curriculum_step": 0, "max_steps": 4 * stage_steps,
            "max_epochs": 10000,  # bounded by max_steps; supports small datasets.
            "validation_every_steps": stage_steps,
            "checkpoint_every_steps": stage_steps,
            "stage4_only_best": True,
            "early_stopping_patience": 100,
        })
        config["data"].update({"max_context_tokens": 1024, "max_atom_tokens": 96,
                               "max_answer_tokens": 64, "max_questions_per_sample": 3})
        config["runtime"].update({"distributed_backend": "none", "reranker_device": "cpu",
                                  "gradient_checkpointing": False, "wandb_mode": "offline",
                                  "checkpoint_upload": "none"})
        validate_config(config, require_runnable=True)
        notebook_config.parent.mkdir(parents=True, exist_ok=True)
        notebook_config.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    if config["runtime"]["distributed_backend"] != "none" or config["training"]["micro_batch_size"] != 1:
        raise ValueError("This notebook requires the real one-document, single-process runtime")
    schedule = [(step, stage_for_step(step, config["training"].get("curriculum_boundaries")).top_k)
                for step in (0, 200, 400, 600)]
    mo.output.append(mo.md(f"## 1-GPU config\n`{notebook_config}` · boundaries/top-K: {schedule} · "
          f"micro-batch=1 × accumulation=16 · max_steps={config['training']['max_steps']}. "
          "FlashAttention-2 is requested with SDPA fallback if unavailable. "
          "A successful CUDA step must be measured; no VRAM guarantee is possible in advance."))
    return config, notebook_config, output_dir


@app.cell
def _(argparse, atoms_path, config, gc, hardware_ok, json, mo, notebook_config,
      output_dir, raw_path, resume_button, torch, train_button, validate_jsonl):

    from doc_to_atom.training.cli import execute as train_execute
    from doc_to_atom.data.io import read_jsonl
    from doc_to_atom.training.loop import run_training

    mo.output.append(mo.md("## Train · run buttons are mutually exclusive; never run concurrently"))
    if train_button.value or resume_button.value:
        if train_button.value and resume_button.value:
            raise ValueError("Choose only one of Train or Resume")
        if not hardware_ok:
            raise RuntimeError("Requires one CUDA BF16 GPU with at least 80 GiB VRAM")
        if not atoms_path.is_file() or not raw_path.is_file():
            raise FileNotFoundError("Source and atomized JSONL are both required")
        _stats = validate_jsonl(atoms_path, source_path=raw_path, reject_span_mismatch=True)
        if _stats["errors"] or _stats["invalid_spans"] or _stats["records"] != sum(1 for _ in raw_path.open(encoding="utf-8")):
            raise ValueError(f"Refusing training on incomplete/invalid data: {_stats}")
        if not any(row.split == "validation" for row in read_jsonl(atoms_path)):
            raise ValueError("Validation split required to select stage-4 best checkpoint")
        pointer = output_dir / "checkpoints/last.json"
        if resume_button.value and not pointer.is_file():
            raise FileNotFoundError(f"No exact-resume checkpoint: {pointer}")
        if train_button.value and (output_dir / "manifests/run.json").exists():
            raise FileExistsError("Existing run present: use Resume; never overwrite run metadata")

        def _args(destination, *, resume=None, steps=None):
            return argparse.Namespace(config=str(notebook_config), data=str(atoms_path),
                output_dir=str(destination), resume=resume, no_reranker=False,
                validate_only=False, max_steps=steps, max_questions=None,
                wandb_mode="offline", wandb_project="doc-to-atom", wandb_entity=None)

        if train_button.value:
            preflight_dir = output_dir.parent / "gemma_preflight"
            if (preflight_dir / "manifests/run.json").exists():
                raise FileExistsError(f"Preflight output exists; verify it before a new run: {preflight_dir}")
            with mo.status.spinner(title="One optimizer-step VRAM preflight"):
                train_execute(_args(preflight_dir, steps=1))
            gc.collect()
            torch.cuda.empty_cache()

        maximum = config["training"]["max_steps"]
        start = 0
        if resume_button.value:
            start = json.loads(pointer.read_text(encoding="utf-8"))["optimizer_step"]
        with mo.status.progress_bar(total=max(0, maximum - start), title="Optimizer updates") as bar:
            def _on_update(event):
                losses = event["train"]["overall"].get("losses", {})
                subtitle = (f"{event['curriculum_stage']} K={event['top_k']} "
                            f"LM={losses.get('lm')} distill={losses.get('distill')} "
                            f"route={losses.get('route')} irrel={losses.get('irrelevant')} "
                            f"protect={losses.get('protect')} sparse={losses.get('sparse')} "
                            f"no-memory={event['train']['generated_irrelevant'].get('no_memory_weight')}")
                bar.update(subtitle=subtitle)

            def _runner(system, tokenizer, records, settings, destination, **kwargs):
                return run_training(system, tokenizer, records, settings, destination,
                                    progress_callback=_on_update, **kwargs)

            training_result = train_execute(
                _args(output_dir, resume="latest" if resume_button.value else None),
                training_runner=_runner,
            )
        mo.output.append(mo.md(f"Training result: `{output_dir / 'reports/training_result.json'}`. "
              f"Optimizer steps: {training_result['training']['optimizer_steps']}. "
              "Check log for non-finite loss/OOM before claiming success."))
    return


@app.cell
def _(argparse, atoms_path, evaluate_button, gc, json, mo, output_dir, raw_path,
      torch, validate_jsonl):
    import hashlib

    from doc_to_atom.data.io import read_jsonl as _read_eval_jsonl
    from doc_to_atom.evaluation.cli import evaluate_checkpoint

    mo.output.append(mo.md("## Validation evaluation"))
    if evaluate_button.value:
        best = output_dir / "checkpoints/best.json"
        snapshot = output_dir / "config/resolved.yaml"
        predictions = output_dir / "predictions/val.jsonl"
        if not best.is_file() or not snapshot.is_file():
            raise FileNotFoundError("Stage-4 best.json and resolved config required before evaluation")
        if not atoms_path.is_file() or not raw_path.is_file():
            raise FileNotFoundError("Evaluation needs matching raw and atomized data")
        _stats = validate_jsonl(atoms_path, source_path=raw_path, reject_span_mismatch=True)
        if _stats["errors"] or _stats["invalid_spans"]:
            raise ValueError(f"Invalid evaluation data: {_stats}")
        unscorable = [qa.question_id for record in _read_eval_jsonl(atoms_path)
                      if record.split == "validation" for qa in record.qa_pairs
                      if not qa.is_irrelevant and not qa.answers]
        if unscorable:
            raise ValueError(f"{len(unscorable)} answerable validation probes lack source gold; "
                             "apply a documented exclusion policy before evaluation")
        if predictions.is_file():
            with predictions.open(encoding="utf-8") as handle:
                first = json.loads(next(line for line in handle if line.strip()))
            if (first.get("source_fingerprint") != hashlib.sha256(raw_path.read_bytes()).hexdigest()
                    or first.get("atomized_fingerprint") != hashlib.sha256(atoms_path.read_bytes()).hexdigest()
                    or first.get("checkpoint_step") != json.loads(best.read_text(encoding="utf-8"))["optimizer_step"]):
                raise ValueError("Existing predictions belong to different data/checkpoint; use a new run path")
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
        args = argparse.Namespace(predictions=str(predictions) if predictions.is_file() else None,
            config=None if predictions.is_file() else str(snapshot),
            data=str(atoms_path), source=str(raw_path),
            checkpoint=None if predictions.is_file() else str(best),
            output=None if predictions.is_file() else str(predictions),
            split="validation", no_reranker=False, wandb_mode="disabled",
            wandb_project="doc-to-atom")
        with mo.status.spinner(title="Generating or scoring validation predictions"):
            _summary = evaluate_checkpoint(args)
        mo.output.append(mo.md(f"Predictions: `{predictions}` · F1={_summary['f1']} · "
              f"ROUGE-L={_summary['rouge_l']} · Refusal F1={_summary['refusal_f1']} · "
              f"latency={_summary['latency_ms']} ms/question"))
    return


@app.cell
def _(json, mo, output_dir):
    import pandas as pd

    from scripts.report_medium_run import summarize

    log = output_dir / "train_log.jsonl"
    metrics_file = output_dir / "metrics/evaluation_validation.json"
    report_file = output_dir / "reports/medium_summary.json"
    if log.is_file() and (output_dir / "config/resolved.yaml").is_file() and metrics_file.is_file():
        if not report_file.is_file():
            _summary = summarize(output_dir, output=report_file)
        else:
            if metrics_file.stat().st_mtime > report_file.stat().st_mtime:
                report_file = output_dir / "reports/medium_summary_refreshed.json"
                if not report_file.is_file():
                    _summary = summarize(output_dir, output=report_file)
                else:
                    _summary = json.loads(report_file.read_text(encoding="utf-8"))
            else:
                _summary = json.loads(report_file.read_text(encoding="utf-8"))
        stages = pd.DataFrame([
            {"stage": stage, "updates": entry["optimizer_updates"],
             "K": entry["top_k"], **entry["loss_means"]}
            for stage, entry in _summary["stages"].items()
        ])
        mo.output.append(mo.md("## Training losses (observed only)"))
        mo.output.append(mo.ui.table(stages))
    else:
        mo.output.append(mo.md("Training report awaits both observed stage logs and validation metrics."))
    if metrics_file.is_file():
        measured = json.loads(metrics_file.read_text(encoding="utf-8"))
        paper_squad = {"F1": 59.71, "ROUGE-L": 59.71, "Refusal F1": 85.25}
        table_1_2 = pd.DataFrame([
            {"dataset": "SQuAD (sample validation)", "metric": key, "paper (%)": paper,
             "Molab (%)": None if measured.get(field) is None else 100 * measured[field],
             "comparable": "No: sample/protocol differ"}
            for key, paper, field in (("F1", paper_squad["F1"], "f1"),
                                       ("ROUGE-L", paper_squad["ROUGE-L"], "rouge_l"),
                                       ("Refusal F1", paper_squad["Refusal F1"], "refusal_f1"))
        ])
        table_3 = pd.DataFrame([
            {"metric": "compile latency, s/update", "paper": 2.10, "Molab": None},
            {"metric": "additional compile memory, GB", "paper": 13.43, "Molab": None},
            {"metric": "evaluation latency, ms/question", "paper": None,
             "Molab": measured.get("latency_ms")},
            {"metric": "evaluation peak allocated VRAM, GiB", "paper": None,
             "Molab": (measured.get("peak_vram_bytes") or 0) / 2**30},
        ])
        mo.output.append(mo.md("## Paper Tables 1–2: measured sample vs published SQuAD (not directly comparable)"))
        mo.output.append(mo.ui.table(table_1_2))
        mo.output.append(mo.md("## Paper Table 3: different measurement scopes, do not conflate"))
        mo.output.append(mo.ui.table(table_3))
    else:
        mo.output.append(mo.md("No validation metrics yet. Table 1–3 measurements are **not available**."))
    return


if __name__ == "__main__":
    app.run()
