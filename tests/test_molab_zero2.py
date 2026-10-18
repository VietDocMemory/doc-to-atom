from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from doc_to_atom.training.molab_gate import (
    MolabGateError,
    RuntimeCapabilities,
    validate_optimizer_shards,
    validate_preflight,
)


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "tests" / "distributed_worker.py"
ACCELERATE_CONFIG = ROOT / "configs" / "accelerate" / "molab_zero2.yaml"
DEEPSPEED_CONFIG = ROOT / "configs" / "deepspeed_zero2.json"
ATOL = 1e-5
RTOL = 1e-4


def _config() -> dict:
    return {
        "training": {
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "effective_global_batch": 4,
        }
    }


def _capabilities(**overrides) -> RuntimeCapabilities:
    values = {
        "gpu_count": 2,
        "nccl_available": True,
        "torch_version": "2.11.0",
        "cuda_version": "12.8",
        "deepspeed_version": "0.18.0",
    }
    values.update(overrides)
    return RuntimeCapabilities(**values)


def test_molab_preflight_records_versions_batch_hashes_and_writable_output(
    tmp_path: Path,
) -> None:
    first = tmp_path / "accelerate.yaml"
    second = tmp_path / "deepspeed.json"
    first.write_text("num_processes: 2\n", encoding="utf-8")
    second.write_text('{"zero_optimization":{"stage":2}}\n', encoding="utf-8")

    report = validate_preflight(
        _config(),
        world_size=2,
        output_dir=tmp_path / "run",
        config_paths=[first, second],
        capabilities=_capabilities(),
    )

    assert report["effective_global_batch"] == 4
    assert report["world_size"] == 2
    assert report["versions"]["deepspeed"] == "0.18.0"
    assert report["config_sha256"][str(first.resolve())] == hashlib.sha256(
        first.read_bytes()
    ).hexdigest()
    assert (tmp_path / "run").is_dir()


def test_molab_preflight_reports_every_blocker_and_batch_mismatch(tmp_path: Path) -> None:
    with pytest.raises(MolabGateError, match="GPU count.*NCCL.*DeepSpeed"):
        validate_preflight(
            _config(),
            world_size=2,
            output_dir=tmp_path,
            config_paths=[],
            capabilities=_capabilities(
                gpu_count=1, nccl_available=False, deepspeed_version=None
            ),
        )
    bad = _config()
    bad["training"]["effective_global_batch"] = 128
    with pytest.raises(MolabGateError, match="effective_global_batch"):
        validate_preflight(
            bad,
            world_size=2,
            output_dir=tmp_path,
            config_paths=[],
            capabilities=_capabilities(),
        )


def test_zero2_optimizer_audit_requires_committed_rank_shards(tmp_path: Path) -> None:
    checkpoint = tmp_path / "step"
    state = checkpoint / "accelerate" / "global_step2"
    state.mkdir(parents=True)
    (checkpoint / "COMMITTED").write_text("distributed-v2\n", encoding="utf-8")
    for rank in range(2):
        (state / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt").write_bytes(b"x")

    report = validate_optimizer_shards(checkpoint, world_size=2)

    assert report["optimizer_shard_count"] == 2
    (state / "bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt").unlink()
    (checkpoint / "accelerate" / "optimizer.bin").write_bytes(b"rank-zero-only")
    with pytest.raises(MolabGateError, match="rank-sharded.*optimizer.bin"):
        validate_optimizer_shards(checkpoint, world_size=2)


def _launch(output: Path, mode: str, resume: Path | None = None) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--config_file",
        str(ACCELERATE_CONFIG),
        "--num_processes",
        "2",
        str(WORKER),
        "--backend",
        "deepspeed_zero2",
        "--mode",
        mode,
        "--output",
        str(output),
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    environment = os.environ.copy()
    environment.update({"WANDB_MODE": "offline", "PYTHONHASHSEED": "0"})
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    (logs / f"{mode}.stdout.log").write_text(completed.stdout, encoding="utf-8")
    (logs / f"{mode}.stderr.log").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode:
        raise AssertionError(
            f"Molab ZeRO-2 {mode} failed with exit code {completed.returncode}; "
            f"logs: {logs}"
        )
    return [
        json.loads((output / f"rank-{rank}.json").read_text(encoding="utf-8"))
        for rank in range(2)
    ]


def _weights(report: dict) -> dict[str, torch.Tensor]:
    return load_file(
        Path(report["checkpoint"]) / "portable" / "model.safetensors", device="cpu"
    )


@pytest.mark.molab
def test_molab_two_gpu_zero2_continuous_and_resume_equivalence(tmp_path: Path) -> None:
    gate = tmp_path / "molab-zero2-gate"
    preflight = validate_preflight(
        _config(),
        world_size=2,
        output_dir=gate,
        config_paths=[ACCELERATE_CONFIG, DEEPSPEED_CONFIG],
    )
    continuous = _launch(gate / "continuous", "continuous")
    interrupted = _launch(gate / "interrupted", "interrupt")
    interrupted_checkpoint = Path(interrupted[0]["checkpoint"])
    optimizer_audit = validate_optimizer_shards(interrupted_checkpoint, world_size=2)
    resumed = _launch(gate / "resumed", "resume", interrupted_checkpoint)

    assert [item["optimizer_step"] for item in resumed] == [2, 2]
    assert [item["curriculum_step"] for item in resumed] == [13, 13]
    assert [item["next_rng"] for item in resumed] == [
        item["next_rng"] for item in continuous
    ]
    expected = _weights(continuous[0])
    actual = _weights(resumed[0])
    for name in expected:
        torch.testing.assert_close(expected[name], actual[name], atol=ATOL, rtol=RTOL)

    reports = gate / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "zero2-equivalence.json").write_text(
        json.dumps(
            {
                "status": "pass",
                "tolerance": {"atol": ATOL, "rtol": RTOL},
                "preflight": preflight,
                "optimizer_audit": optimizer_audit,
                "continuous_checkpoint": continuous[0]["checkpoint"],
                "resumed_checkpoint": resumed[0]["checkpoint"],
                "wandb_mode": "offline",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
