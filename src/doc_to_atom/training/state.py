from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class TrainerState:
    """Serializable control state at an optimizer-step checkpoint boundary."""

    schema_version: int = 2
    epoch: int = 0
    record_cursor: int = 0
    micro_step: int = 0
    optimizer_step: int = 0
    curriculum_step: int = 0
    sampler_state: dict[str, Any] = field(default_factory=dict)
    best_validation: float = float("inf")
    bad_validations: int = 0
    validation_observations: int = 0
    early_stopped: bool = False
    config_fingerprint: str = ""
    dataset_fingerprint: str = ""
    model_fingerprint: str = ""
    prompt_fingerprint: str = ""
    wandb_run_id: str | None = None
    wandb_mode: str = "disabled"
    world_size: int = 1
    distributed_backend: str = "none"
    exact_resume: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TrainerState":
        return cls(**value)

