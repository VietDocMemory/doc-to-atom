from importlib.util import module_from_spec, spec_from_file_location
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
