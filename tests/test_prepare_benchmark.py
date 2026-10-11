from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "prepare_benchmark.py"
    spec = spec_from_file_location("prepare_benchmark", path)
    module = module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_groups_squad_questions_sharing_a_context():
    module = _module()
    records = module.prepare_rows(
        [
            {"id": "a", "context": "Context", "question": "First?", "answers": {"text": ["one"]}},
            {"id": "b", "context": "Context", "question": "Second?", "answers": {"text": ["two"]}},
        ],
        "squad",
        "test",
        "squad",
    )
    assert len(records) == 1
    assert records[0]["split"] == "test"
    assert [question["id"] for question in records[0]["questions"]] == ["a", "b"]
