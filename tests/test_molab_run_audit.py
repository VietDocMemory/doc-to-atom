import importlib.util
import json
from pathlib import Path

import yaml

spec = importlib.util.spec_from_file_location(
    "molab_audit", Path(__file__).parents[1] / "scripts/audit_molab_run.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_audit_is_read_only_and_identifies_legacy_protocol_mismatch(tmp_path):
    (tmp_path / "config").mkdir()
    config = {"training": {"warmup_steps": 500, "max_steps": 800}}
    (tmp_path / "config/resolved.yaml").write_text(yaml.safe_dump(config))
    checkpoint = tmp_path / "checkpoints/step-00000800"
    checkpoint.mkdir(parents=True)
    (checkpoint / "state.json").write_text(json.dumps({"optimizer_step": 800, "dataset_fingerprint": ""}))
    pointer = json.dumps({"optimizer_step": 800, "path": "step-00000800"})
    (tmp_path / "checkpoints/best.json").write_text(pointer)
    (tmp_path / "checkpoints/last.json").write_text(pointer)
    groups = {name: {"count": count, "losses": {"lm": loss}}
              for name, count, loss in (("overall", 299, 2.5), ("answerable", 100, 5.7),
                                        ("generated_irrelevant", 199, 0.9))}
    row = {"optimizer_step": 800, "curriculum_stage": "full_composition",
           "learning_rate": 0, "gradient_norm": 25, "validation": groups}
    (tmp_path / "train_log.jsonl").write_text(json.dumps(row) + "\n")
    before = {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    report = module.audit(tmp_path)
    assert report["completed_optimizer_steps"] == 800
    assert report["selected_validation"]["answerable_count"] == 100
    assert report["selected_validation"]["irrelevant_count"] == 199
    assert "TRAIN_INFERENCE_PROMPT_MISMATCH" in {item["code"] for item in report["findings"]}
    assert report["nonfinite_optimizer_steps"] == []
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
