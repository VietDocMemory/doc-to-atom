from pathlib import Path

import json
import pytest

from doc_to_atom.config import load_and_resolve_config
from doc_to_atom.data import prepare as dataset_prepare
from doc_to_atom.training.curriculum import stage_for_step
from scripts.prepare_medium_config import prepare
from scripts.report_medium_run import summarize


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("steps", [200, 500])
def test_medium_config_is_one_continuous_run_with_shortened_stage_boundaries(tmp_path, steps):
    path = tmp_path / "medium.yaml"
    prepare(ROOT / "configs/reconstruction/molab_single_gpu.yaml", path,
            stage_steps=steps, train_records=500)
    config = load_and_resolve_config(path, require_runnable=True)
    training = config["training"]
    assert training["curriculum_boundaries"] == [steps, 2 * steps, 3 * steps]
    assert training["max_steps"] == 4 * steps
    assert training["micro_batch_size"] == 1
    assert training["gradient_accumulation_steps"] == 16
    assert training["effective_global_batch"] == 16
    assert training["stage4_only_best"] is True
    for step, expected in ((0, 1), (steps - 1, 1), (steps, 4),
                           (2 * steps, 8), (3 * steps, 8)):
        assert stage_for_step(step, training["curriculum_boundaries"]).top_k == expected
    assert stage_for_step(2000).top_k == 4  # Paper schedule remains the default.
    assert any(item["id"] == "LOCAL-MEDIUM-CURRICULUM" for item in config["deviations"])
    with pytest.raises(FileExistsError):
        prepare(ROOT / "configs/reconstruction/molab_single_gpu.yaml", path,
                stage_steps=steps, train_records=500)


@pytest.mark.parametrize("boundaries", [[0, 200, 400], [200, 200, 400],
                                         [200, 400], [200, "400", 600]])
def test_invalid_medium_boundaries_are_rejected(boundaries):
    with pytest.raises(ValueError, match="curriculum_boundaries"):
        stage_for_step(0, boundaries)


def test_squad_streaming_counts_distinct_context_records_not_question_rows(tmp_path, monkeypatch):
    def fake_load(name, *, split, streaming):
        assert name == "rajpurkar/squad" and streaming
        for index, context in enumerate(("A", "A", "B", "C", "D")):
            yield {"id": f"{split}-{index}", "context": context, "question": "q",
                   "answers": {"text": ["A"]}}

    monkeypatch.setattr(dataset_prepare, "load_dataset", fake_load)
    path = tmp_path / "raw.jsonl"
    stats = dataset_prepare.prepare_squad_records(path, train_records=2, validation_records=1)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert stats["records"] == len(rows) == 3
    assert [row["context"] for row in rows] == ["A", "B", "A"]
    assert [row["split"] for row in rows] == ["train", "train", "validation"]
    with pytest.raises(FileExistsError):
        dataset_prepare.prepare_squad_records(path, 2, 1)


def test_medium_report_uses_only_observed_stages_and_metrics(tmp_path):
    run = tmp_path / "run"
    with pytest.raises(FileNotFoundError):
        summarize(run, output=tmp_path / "missing.json")
    (run / "config").mkdir(parents=True)
    config = prepare(ROOT / "configs/reconstruction/molab_single_gpu.yaml",
                     run / "config/resolved.yaml", stage_steps=200, train_records=500)
    assert config["training"]["max_steps"] == 800
    stages = ("single_atom", "multi_atom", "robustness", "full_composition")
    with (run / "train_log.jsonl").open("w", encoding="utf-8") as handle:
        for step, stage in zip((200, 400, 600, 800), stages, strict=True):
            handle.write(json.dumps({
                "optimizer_step": step, "curriculum_stage": stage,
                "top_k": 1 if step == 200 else 4 if step == 400 else 8,
                "train": {"overall": {"losses": {"lm": 1.0, "route": 0.5}}},
            }) + "\n")
    report = summarize(run, output=tmp_path / "summary.json")
    assert report["training_completed"]
    assert report["evaluation"] is None
    assert report["stages"]["full_composition"]["top_k"] == [8]
    assert report["stages"]["single_atom"]["loss_means"]["distill"] is None
