from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file


ATOL = 1e-7
RTOL = 1e-6
WORKER = Path(__file__).with_name("distributed_worker.py")


def _environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "ACCELERATE_USE_CPU": "true",
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
            "USE_LIBUV": "0",
        }
    )
    return environment


def _run(command: list[str], *, timeout: float = 90.0) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_environment(),
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise AssertionError(
            f"distributed subprocess timed out after {timeout}s: {' '.join(command)}\n"
            f"stdout:\n{exc.stdout or ''}\nstderr:\n{exc.stderr or ''}"
        ) from exc
    if completed.returncode:
        raise AssertionError(
            f"distributed subprocess/rank crash (exit code {completed.returncode}): "
            f"{' '.join(command)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed


def _launch(
    output: Path,
    *,
    processes: int,
    mode: str,
    resume: Path | None = None,
    crash_rank: int | None = None,
) -> list[dict]:
    if resume is not None and not (resume / "COMMITTED").is_file():
        raise AssertionError(f"partial distributed checkpoint lacks COMMITTED: {resume}")
    command = [
        sys.executable,
        str(WORKER),
        "--mode",
        mode,
        "--output",
        str(output),
    ]
    if resume is not None:
        command.extend(("--resume", str(resume)))
    if crash_rank is not None:
        command.extend(("--crash-rank", str(crash_rank)))
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        master_port = str(listener.getsockname()[1])
    children: list[subprocess.Popen[str]] = []
    for rank in range(processes):
        environment = _environment()
        environment.update(
            {
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": master_port,
                "RANK": str(rank),
                "WORLD_SIZE": str(processes),
                "LOCAL_RANK": str(rank),
                "LOCAL_WORLD_SIZE": str(processes),
            }
        )
        children.append(
            subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=environment,
            )
        )
    deadline = time.monotonic() + 90.0
    while True:
        failed_rank = next(
            (rank for rank, child in enumerate(children) if (child.poll() or 0) != 0), None
        )
        if failed_rank is not None or all(child.poll() is not None for child in children):
            break
        if time.monotonic() >= deadline:
            for child in children:
                child.terminate()
            outputs = [child.communicate(timeout=5) for child in children]
            raise AssertionError(
                "distributed subprocess timed out; "
                + "; ".join(
                    f"rank {rank} stdout:\n{stdout}\nstderr:\n{stderr}"
                    for rank, (stdout, stderr) in enumerate(outputs)
                )
            )
        time.sleep(0.02)
    if failed_rank is not None:
        for rank, child in enumerate(children):
            if rank != failed_rank and child.poll() is None:
                child.terminate()
        outputs = [child.communicate(timeout=5) for child in children]
        raise AssertionError(
            f"distributed subprocess/rank crash (rank {failed_rank}, exit code "
            f"{children[failed_rank].returncode}); "
            + "; ".join(
                f"rank {rank} stdout:\n{stdout}\nstderr:\n{stderr}"
                for rank, (stdout, stderr) in enumerate(outputs)
            )
        )
    for child in children:
        child.communicate(timeout=5)
    reports = []
    for rank in range(processes):
        report_path = output / f"rank-{rank}.json"
        if not report_path.is_file():
            raise AssertionError(f"rank {rank} completed without report artifact: {report_path}")
        reports.append(json.loads(report_path.read_text(encoding="utf-8")))
    return reports


def _consumed(*phases: list[dict]) -> list[int]:
    entries = [item for reports in phases for report in reports for item in report["consumed"]]
    return [item["sample_id"] for item in sorted(entries, key=lambda value: value["position"])]


def _portable_weights(report: dict) -> dict[str, torch.Tensor]:
    checkpoint = Path(report["checkpoint"])
    assert (checkpoint / "COMMITTED").is_file()
    return load_file(checkpoint / "portable" / "model.safetensors", device="cpu")


def _assert_nested_close(left, right) -> None:
    if isinstance(left, list):
        assert isinstance(right, list) and len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_close(left_item, right_item)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_close(left[key], right[key])
    elif isinstance(left, float):
        assert left == pytest.approx(right, abs=ATOL, rel=RTOL)
    else:
        assert left == right


@pytest.mark.integration
@pytest.mark.distributed
def test_one_process_and_two_process_gloo_are_numerically_equivalent(tmp_path: Path) -> None:
    single = _launch(tmp_path / "single", processes=1, mode="continuous")
    distributed = _launch(tmp_path / "distributed", processes=2, mode="continuous")

    assert _consumed(single) == _consumed(distributed) == [4, 6, 3, 0, 5, 2, 1]
    assert single[0]["optimizer_step"] == distributed[0]["optimizer_step"] == 2
    assert single[0]["curriculum_step"] == distributed[0]["curriculum_step"] == 13
    assert single[0]["learning_rate"] == pytest.approx(
        distributed[0]["learning_rate"], abs=ATOL, rel=RTOL
    )
    single_weights = _portable_weights(single[0])
    distributed_weights = _portable_weights(distributed[0])
    assert single_weights.keys() == distributed_weights.keys()
    for name in single_weights:
        torch.testing.assert_close(
            single_weights[name], distributed_weights[name], atol=ATOL, rtol=RTOL
        )


@pytest.mark.integration
@pytest.mark.distributed
def test_interrupted_two_process_resume_matches_continuous_state(tmp_path: Path) -> None:
    continuous = _launch(tmp_path / "continuous", processes=2, mode="continuous")
    interrupted = _launch(tmp_path / "interrupted", processes=2, mode="interrupt")
    resume_checkpoint = Path(interrupted[0]["checkpoint"])
    accelerate_state = resume_checkpoint / "accelerate"
    assert (resume_checkpoint / "COMMITTED").is_file()
    assert (accelerate_state / "optimizer.bin").is_file()
    assert (accelerate_state / "scheduler.bin").is_file()
    assert (accelerate_state / "random_states_0.pkl").is_file()
    assert (accelerate_state / "random_states_1.pkl").is_file()
    resumed = _launch(
        tmp_path / "resumed", processes=2, mode="resume", resume=resume_checkpoint
    )

    assert _consumed(interrupted, resumed) == _consumed(continuous)
    for continuous_rank, resumed_rank in zip(continuous, resumed, strict=True):
        assert resumed_rank["optimizer_step"] == continuous_rank["optimizer_step"] == 2
        assert resumed_rank["curriculum_step"] == continuous_rank["curriculum_step"] == 13
        assert resumed_rank["record_cursor"] == continuous_rank["record_cursor"] == 0
        assert resumed_rank["sampler_cursor"] == continuous_rank["sampler_cursor"] == 4
        assert resumed_rank["learning_rate"] == pytest.approx(
            continuous_rank["learning_rate"], abs=ATOL, rel=RTOL
        )
        _assert_nested_close(resumed_rank["optimizer"], continuous_rank["optimizer"])
        _assert_nested_close(resumed_rank["next_rng"], continuous_rank["next_rng"])
    continuous_weights = _portable_weights(continuous[0])
    resumed_weights = _portable_weights(resumed[0])
    for name in continuous_weights:
        torch.testing.assert_close(
            continuous_weights[name], resumed_weights[name], atol=ATOL, rtol=RTOL
        )


def test_harness_reports_timeout_rank_crash_and_partial_checkpoint(tmp_path: Path) -> None:
    with pytest.raises(AssertionError, match=r"(?s)timed out.*stdout.*stderr"):
        _run([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.05)

    with pytest.raises(AssertionError, match="rank crash.*exit code"):
        _launch(tmp_path / "crash", processes=2, mode="continuous", crash_rank=1)

    partial = tmp_path / "partial"
    partial.mkdir()
    with pytest.raises(AssertionError, match="lacks COMMITTED"):
        _launch(tmp_path / "unused", processes=2, mode="resume", resume=partial)
