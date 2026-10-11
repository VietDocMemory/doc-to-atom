from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import zipfile

from doc_to_atom.data.benchmark_catalog import get_benchmark_dataset


def _module():
    path = Path(__file__).parents[1] / "scripts" / "prepare_d2a_benchmark.py"
    spec = spec_from_file_location("prepare_d2a_benchmark", path)
    module = module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_longbench_loader_reads_jsonl_without_dataset_script(tmp_path, monkeypatch):
    module = _module()
    archive = tmp_path / "data.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("data/qasper.jsonl", '{"context":"c","input":"q","answers":["a"]}\n')
    monkeypatch.setattr(module, "hf_hub_download", lambda **_kwargs: archive)
    assert module._load_longbench_rows(get_benchmark_dataset("longbench_qasper"), None) == [
        {"context": "c", "input": "q", "answers": ["a"]}
    ]
