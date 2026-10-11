from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import pytest


def _module():
    path = Path(__file__).parents[1] / "scripts" / "run_d2a_benchmark.py"
    spec = spec_from_file_location("run_d2a_benchmark", path)
    module = module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_runner_reports_missing_atomized_dataset_file(tmp_path):
    module = _module()
    with pytest.raises(FileNotFoundError, match="atomize.py"):
        module.load_named_records(tmp_path, ["squad"], "test")


def test_runner_allows_invalid_spans_only_when_explicit(tmp_path):
    module = _module()
    source = Path(__file__).parents[1] / "data" / "squad_small_atoms_openai.jsonl"
    row = json.loads(source.read_text(encoding="utf-8").splitlines()[0])
    row["split"] = "test"
    row["atoms"][0]["source_span"] = "this span is absent"
    (tmp_path / "squad_test.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source spans"):
        module.load_named_records(tmp_path, ["squad"], "test")
    records, stats = module.load_named_records(tmp_path, ["squad"], "test", allow_invalid_spans=True)
    assert len(records) == 1
    assert stats[0]["invalid_spans"] >= 1
