from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from doc_to_atom.credentials import redact_secrets

ROOT = Path(__file__).resolve().parents[2]

CONFIG_FAMILIES = frozenset({"paper_like", "reconstruction", "experiments", "runtime"})
ASSUMPTION_FIELDS = frozenset(
    {
        "id",
        "category",
        "description",
        "chosen_value",
        "rationale",
        "affected_metrics",
        "source_status",
    }
)
DEVIATION_FIELDS = frozenset({"id", "description", "rationale", "affected_metrics"})

# These values select an execution environment or an equivalent implementation;
# they do not change the learned model contract.  Keeping them out of the
# fingerprint lets a checkpoint trained on one host be evaluated on another.
_FINGERPRINT_IGNORED_KEYS = frozenset(
    {
        "attention_implementation",
        "attn_implementation",
        "cache_dir",
        "checkpoint_dir",
        "device",
        "device_map",
        "flash_attention",
        "flash_attention_2",
        "log_dir",
        "logging_dir",
        "output_dir",
        "reranker_device",
        "run_dir",
        "use_flash_attention_2",
        "wandb_dir",
    }
)


class ConfigError(ValueError):
    """Raised when configuration provenance or invariants are invalid."""


def load_project_env() -> None:
    """Load local credentials without returning or logging their values."""
    load_dotenv(ROOT / ".env", override=False)


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if key == "_base_":
            continue
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_config(path: Path, ancestry: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()
    if path in ancestry:
        chain = " -> ".join(str(item) for item in (*ancestry, path))
        raise ConfigError(f"configuration inheritance cycle: {chain}")
    if not path.is_file():
        raise ConfigError(f"configuration file does not exist: {path}")
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    if not isinstance(config, dict):
        raise ConfigError(f"configuration root must be a mapping: {path}")
    if base_name := config.get("_base_"):
        if not isinstance(base_name, str):
            raise ConfigError(f"_base_ must be a path string: {path}")
        base = _load_config(path.parent / base_name, (*ancestry, path))
        config = _merge(base, config)
    return config


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a recursively inherited YAML config without schema enforcement."""
    return _load_config(Path(path), ())


def canonical_config_json(config: dict[str, Any]) -> str:
    """Serialize a resolved config deterministically for hashing and snapshots."""
    return json.dumps(
        config,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _fingerprint_config(value: Any) -> Any:
    """Return a copy without host-specific execution settings."""
    if isinstance(value, dict):
        return {
            key: _fingerprint_config(item)
            for key, item in value.items()
            if str(key).lower().replace("-", "_") not in _FINGERPRINT_IGNORED_KEYS
        }
    if isinstance(value, list):
        return [_fingerprint_config(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_fingerprint_config(item) for item in value)
    return value


def config_fingerprint(config: dict[str, Any]) -> str:
    semantic_config = _fingerprint_config(redact_secrets(config))
    return hashlib.sha256(canonical_config_json(semantic_config).encode("utf-8")).hexdigest()


def legacy_config_fingerprint(config: dict[str, Any]) -> str:
    """Hash the complete config for compatibility with pre-normalization checkpoints."""
    return hashlib.sha256(canonical_config_json(config).encode("utf-8")).hexdigest()


def config_fingerprint_candidates(config: dict[str, Any]) -> set[str]:
    """Strict hashes of this config under current and historical algorithms.

    Historical credential-bearing checkpoints require the original config, not
    guessed values from a corrupted snapshot. Checkpoint metadata is never edited.
    """
    historical_semantic = hashlib.sha256(
        canonical_config_json(_fingerprint_config(config)).encode("utf-8")
    ).hexdigest()
    return {config_fingerprint(config), historical_semantic, legacy_config_fingerprint(config)}


def _validate_metadata_entries(
    entries: Any, *, kind: str, required_fields: frozenset[str]
) -> None:
    if not isinstance(entries, list):
        raise ConfigError(f"{kind}s must be a list")
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ConfigError(f"{kind}[{index}] must be a mapping")
        entry_id = str(entry.get("id", f"at index {index}"))
        missing = sorted(required_fields - entry.keys())
        if missing:
            raise ConfigError(f"{kind} {entry_id} missing fields: {', '.join(missing)}")
        if entry_id in seen:
            raise ConfigError(f"duplicate {kind} id: {entry_id}")
        seen.add(entry_id)


def validate_config(config: dict[str, Any], *, require_runnable: bool = False) -> None:
    """Validate classification/provenance and optionally all runnable training values."""
    profile = config.get("profile")
    if not isinstance(profile, dict) or not profile.get("name"):
        raise ConfigError("profile.name is required")
    family = profile.get("family")
    if family not in CONFIG_FAMILIES:
        raise ConfigError(
            f"profile.family must be one of {sorted(CONFIG_FAMILIES)}, got {family!r}"
        )
    _validate_metadata_entries(
        config.get("assumptions", []), kind="assumption", required_fields=ASSUMPTION_FIELDS
    )
    _validate_metadata_entries(
        config.get("deviations", []), kind="deviation", required_fields=DEVIATION_FIELDS
    )
    training = config.get("training", {})
    runtime = config.get("runtime", {})
    if training.get("checkpoint_retention", "best_last") != "best_last":
        raise ConfigError("training.checkpoint_retention must be best_last")
    interval = training.get("checkpoint_every_steps", 1000)
    if type(interval) is not int or interval < 1:
        raise ConfigError("training.checkpoint_every_steps must be a positive integer")
    if runtime.get("checkpoint_upload", "best_weights") not in {
        "best_weights", "best_weights_and_last_full", "none"
    }:
        raise ConfigError(
            "runtime.checkpoint_upload must be best_weights, best_weights_and_last_full, or none"
        )
    if require_runnable:
        try:
            require_disclosed_training_values(config)
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(str(exc)) from exc


def load_and_resolve_config(
    path: str | Path, *, require_runnable: bool = False
) -> dict[str, Any]:
    config = load_config(path)
    validate_config(config, require_runnable=require_runnable)
    return config


def require_disclosed_training_values(config: dict[str, Any]) -> None:
    training = config["training"]
    missing = [name for name, value in training["loss_weights"].items() if value is None]
    if training.get("sparse_target_density") is None:
        missing.append("sparse_target_density/rho")
    if missing:
        joined = ", ".join(missing)
        raise ValueError(
            f"PAPER_UNSPECIFIED values must be supplied before training: {joined}. "
            "Use a separate assumption config; do not alter the paper config silently."
        )


def write_run_manifest(config: dict[str, Any], output: str | Path) -> None:
    output = Path(output)
    deviations: list[str] = []
    runtime = config.get("runtime", {})
    if runtime.get("profile") == "resource_constrained":
        deviations.extend(
            [
                "single-GPU/resource-constrained profile",
                "effective batch differs from paper"
                if config["training"]["effective_global_batch"] != 128
                else "",
                "loss coefficients and rho are assumptions because the paper does not publish them",
                "record-step limit for smoke execution"
                if config["training"].get("max_steps") is not None
                else "",
                "question count limited for smoke execution"
                if config["data"].get("max_questions_per_sample", 8) < 8
                else "",
                "warmup disabled for short overfit diagnostic"
                if config["training"].get("warmup_steps") == 0
                else "",
                "curriculum starts at a later paper stage so K>1 routing is observable"
                if config["training"].get("initial_curriculum_step", 0) > 0
                else "",
                "periodic validation is enabled for the short diagnostic"
                if config["training"].get("validation_every_steps")
                else "",
            ]
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"config": config, "reproduction_deviations": [item for item in deviations if item]},
            indent=2,
        ),
        encoding="utf-8",
    )
