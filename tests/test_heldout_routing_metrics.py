from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _module():
    path = Path(__file__).parents[1] / "scripts" / "run_heldout_generalization.py"
    spec = spec_from_file_location("run_heldout_generalization", path)
    module = module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_routing_metrics_are_undefined_without_gold_atoms():
    module = _module()
    assert module._routing_metrics(["atom_1"], []) == (None, None, None, None)
