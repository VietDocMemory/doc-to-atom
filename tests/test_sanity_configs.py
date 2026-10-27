from pathlib import Path

import pytest

from doc_to_atom.config import load_and_resolve_config
from doc_to_atom.training.curriculum import stage_for_step
from scripts.prepare_sanity_configs import prepare


ROOT = Path(__file__).resolve().parents[1]


def test_generated_sanity_profiles_exercise_four_boundaries_without_changing_base(tmp_path):
    base = ROOT / "configs" / "reconstruction" / "molab_single_gpu.yaml"
    original = base.read_bytes()
    paths = prepare(base, tmp_path, steps=2)
    assert [path.name for path in paths] == [
        "stage1.yaml", "stage2.yaml", "stage3.yaml", "stage4.yaml"
    ]
    assert base.read_bytes() == original
    for path, expected_start, expected_k in zip(
        paths, (0, 2000, 6000, 9000), (1, 4, 8, 8), strict=True
    ):
        config = load_and_resolve_config(path, require_runnable=True)
        assert config["profile"]["family"] == "experiments"
        assert config["training"]["initial_curriculum_step"] == expected_start
        assert stage_for_step(expected_start).top_k == expected_k
        assert config["training"]["max_steps"] == 2
        assert config["training"]["effective_global_batch"] == 1
        assert any(item["id"].startswith("CURRICULUM-DIAGNOSTIC")
                   for item in config["deviations"])
    with pytest.raises(FileExistsError):
        prepare(base, tmp_path)


def test_blackwell_single_gpu_config_does_not_claim_eight_gpu_batch():
    config = load_and_resolve_config(
        ROOT / "configs" / "reconstruction" / "molab_blackwell_1gpu.yaml",
        require_runnable=True,
    )
    assert config["runtime"]["distributed_backend"] == "none"
    assert config["training"]["micro_batch_size"] == 1
    assert config["training"]["gradient_accumulation_steps"] == 16
    assert config["training"]["effective_global_batch"] == 16
