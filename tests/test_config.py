from __future__ import annotations

from pathlib import Path

import pytest

import doc_to_atom.config as config_module


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_recursive_base_resolution_merges_without_control_keys(tmp_path: Path) -> None:
    _write(
        tmp_path / "base.yaml",
        "profile: {name: base, family: paper_like}\n"
        "assumptions: []\ndeviations: []\n"
        "training: {learning_rate: 0.001, max_epochs: 2}\n",
    )
    _write(
        tmp_path / "middle.yaml",
        "_base_: base.yaml\nprofile: {name: middle, family: reconstruction}\n"
        "training: {max_epochs: 3}\n",
    )
    leaf = _write(
        tmp_path / "leaf.yaml",
        "_base_: middle.yaml\nprofile: {name: leaf, family: experiments}\n"
        "training: {learning_rate: 0.0002}\n",
    )

    resolved = config_module.load_and_resolve_config(leaf)

    assert resolved["training"] == {"learning_rate": 0.0002, "max_epochs": 3}
    assert resolved["profile"] == {"name": "leaf", "family": "experiments"}
    assert "_base_" not in resolved


def test_recursive_base_resolution_rejects_cycles(tmp_path: Path) -> None:
    first = _write(tmp_path / "first.yaml", "_base_: second.yaml\n")
    _write(tmp_path / "second.yaml", "_base_: first.yaml\n")

    with pytest.raises(config_module.ConfigError, match="configuration inheritance cycle"):
        config_module.load_and_resolve_config(first)


def test_runnable_validation_rejects_undisclosed_training_values() -> None:
    config = {
        "profile": {"name": "paper", "family": "paper_like"},
        "assumptions": [],
        "deviations": [],
        "training": {
            "sparse_target_density": None,
            "loss_weights": {"lm": None, "route": 1.0},
        },
    }

    with pytest.raises(
        config_module.ConfigError, match=r"(?=.*\blm\b)(?=.*sparse_target_density/rho)"
    ):
        config_module.validate_config(config, require_runnable=True)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("profile", {"name": "bad", "family": "unknown"}, "profile.family"),
        (
            "assumptions",
            [{"id": "A1", "category": "training"}],
            "assumption A1 missing",
        ),
        ("deviations", [{"id": "D1"}], "deviation D1 missing"),
    ],
)
def test_config_metadata_is_structurally_validated(field, value, message) -> None:
    config = {
        "profile": {"name": "valid", "family": "reconstruction"},
        "assumptions": [],
        "deviations": [],
        field: value,
    }

    with pytest.raises(config_module.ConfigError, match=message):
        config_module.validate_config(config)


def test_canonical_serialization_and_fingerprint_ignore_mapping_order() -> None:
    left = {
        "profile": {"name": "run", "family": "experiments"},
        "assumptions": [],
        "deviations": [],
        "training": {"learning_rate": 0.001, "betas": [0.9, 0.999]},
    }
    right = {
        "training": {"betas": [0.9, 0.999], "learning_rate": 0.001},
        "deviations": [],
        "assumptions": [],
        "profile": {"family": "experiments", "name": "run"},
    }

    assert config_module.canonical_config_json(left) == config_module.canonical_config_json(right)
    assert config_module.config_fingerprint(left) == config_module.config_fingerprint(right)
    assert len(config_module.config_fingerprint(left)) == 64


def test_checked_in_config_families_are_classified_and_runnable() -> None:
    root = Path(__file__).resolve().parents[1]
    paper = config_module.load_and_resolve_config(root / "configs/paper_like/base.yaml")
    reconstruction = config_module.load_and_resolve_config(
        root / "configs/reconstruction/molab_zero2.yaml", require_runnable=True
    )
    experiment = config_module.load_and_resolve_config(
        root / "configs/experiments/small_overfit.yaml", require_runnable=True
    )
    runtime = config_module.load_and_resolve_config(root / "configs/runtime/local_offline.yaml")

    assert paper["profile"]["family"] == "paper_like"
    assert reconstruction["profile"]["family"] == "reconstruction"
    assert experiment["profile"]["family"] == "experiments"
    assert runtime["profile"]["family"] == "runtime"
    assert reconstruction["compiler"]["micro_kv"]["enabled"] is False
    assert reconstruction["runtime"]["gradient_checkpointing"] is False
    assert {item["id"] for item in reconstruction["assumptions"]}

