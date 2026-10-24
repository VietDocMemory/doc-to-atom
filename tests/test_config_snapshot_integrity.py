from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

import doc_to_atom.config as config_module
from doc_to_atom.config import canonical_config_json, config_fingerprint, load_config
from doc_to_atom.runtime import RunContext, RunTracker
from doc_to_atom.runtime import run as run_module
from doc_to_atom.training.checkpoint import (
    IncompatibleCheckpointError, load_training_checkpoint, save_training_checkpoint,
)
from doc_to_atom.training.state import TrainerState


def _config():
    return {
        "profile": {"name": "snapshot-test", "family": "experiments"},
        "training": {"learning_rate": 0.001},
        "data": {"max_context_tokens": 8192, "max_atom_tokens": 96, "max_answer_tokens": 128},
        "evaluation": {"max_new_tokens": 8},
        "tokenizer": {"name": "example", "bos_token": "<bos>", "eos_token_id": 2},
    }


@pytest.mark.parametrize("credentials", [False, True])
def test_snapshot_preserves_semantics_and_fingerprint(tmp_path, credentials):
    config = _config()
    if credentials:
        config["provider"] = {"api_key": "private-api-value", "HF_TOKEN": "private-hf-value"}
    before = deepcopy(config)
    context = RunContext.create(tmp_path / "run", config=config, command=["train"])
    snapshot = load_config(context.root / "config/resolved.yaml")
    assert snapshot["data"] == config["data"]
    assert snapshot["evaluation"] == config["evaluation"]
    assert snapshot["tokenizer"] == config["tokenizer"]
    assert config_fingerprint(snapshot) == config_fingerprint(config) == context.config_fingerprint
    assert config == before
    if credentials:
        assert snapshot["provider"] == {"api_key": "<redacted>", "HF_TOKEN": "<redacted>"}


@pytest.mark.parametrize("key", [
    "api_key", "api-key", "OPENAI_API_KEY", "apiKey", "HF_TOKEN", "access_token",
    "refreshToken", "password", "db_password", "client_secret", "AWS_SECRET_ACCESS_KEY",
    "credentials", "Authorization", "private_key",
])
def test_credentials_are_redacted(key):
    assert run_module.redact_secrets({"nested": [{key: "private-value"}]}) == {
        "nested": [{key: "<redacted>"}]
    }


def test_cli_credentials_are_redacted_without_losing_numeric_arguments():
    command = ["train", "--max-new-tokens", "8", "--api-key=private-inline",
               "--access-token", "private-separated"]
    assert run_module.redact_secrets(command) == [
        "train", "--max-new-tokens", "8", "--api-key=<redacted>",
        "--access-token", "<redacted>",
    ]
    assert run_module.redact_secrets(tuple(command)) == tuple(run_module.redact_secrets(command))


def test_semantic_token_lists_are_not_interpreted_as_cli_arguments(tmp_path):
    config = _config()
    config["tokenizer"]["additional_special_tokens"] = ["--api-key", "<ordinary-token>"]
    context = RunContext.create(tmp_path / "run", config=config, command=["train"])
    snapshot = load_config(context.root / "config/resolved.yaml")
    assert snapshot["tokenizer"] == config["tokenizer"]
    assert config_fingerprint(snapshot) == config_fingerprint(config)


def test_snapshot_corruption_fails_before_run_publication(tmp_path, monkeypatch):
    original_write = run_module.atomic_write_yaml

    def corrupt_write(path, config):
        changed = deepcopy(config)
        changed["training"]["learning_rate"] = 99
        original_write(path, changed)

    monkeypatch.setattr(run_module, "atomic_write_yaml", corrupt_write)
    with pytest.raises(run_module.RunContextError, match="snapshot.*fingerprint"):
        RunContext.create(tmp_path / "run", config=_config(), command=["train"])
    assert not (tmp_path / "run/manifests/run.json").exists()


def test_training_refuses_changed_snapshot_before_model_allocation(tmp_path, monkeypatch):
    from doc_to_atom.training import cli

    original_write = run_module.atomic_write_yaml

    def corrupt_write(path, config):
        changed = deepcopy(config)
        changed["training"]["learning_rate"] *= 2
        original_write(path, changed)

    monkeypatch.setattr(run_module, "atomic_write_yaml", corrupt_write)
    args = cli.build_parser().parse_args([
        "--config", str(Path(__file__).parents[1] / "configs/experiments/small_overfit.yaml"),
        "--data", str(tmp_path / "unused.jsonl"), "--output", str(tmp_path / "run"),
        "--wandb-mode", "disabled",
    ])
    with pytest.raises(run_module.RunContextError, match="snapshot.*fingerprint"):
        cli.execute(args, system_factory=lambda *a, **kw: pytest.fail("model allocated"),
                    training_runner=lambda *a, **kw: pytest.fail("training started"))


def test_resume_rejects_corrupt_snapshot(tmp_path):
    context = RunContext.create(tmp_path / "run", config=_config(), command=["train"])
    snapshot = context.root / "config/resolved.yaml"
    config = load_config(snapshot)
    config["evaluation"]["max_new_tokens"] = 99
    snapshot.write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(run_module.RunContextError, match="snapshot.*fingerprint"):
        RunContext.create(context.root, config=_config(), command=["train"], resume=True)


def test_snapshot_checkpoint_roundtrip_and_changed_semantics_rejected(tmp_path):
    config = _config()
    config["provider"] = {"api_key": "private-value"}
    context = RunContext.create(tmp_path / "run", config=config, command=["train"])
    module = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(module.parameters(), lr=config["training"]["learning_rate"])
    module(torch.ones(1, 2)).sum().backward()
    optimizer.step()
    checkpoint = context.root / "checkpoints/step-00000001"
    save_training_checkpoint({"module": module}, checkpoint,
        state=TrainerState(optimizer_step=1, config_fingerprint=config_fingerprint(config)))
    snapshot = load_config(context.root / "config/resolved.yaml")
    loaded = torch.nn.Linear(2, 1)
    state = load_training_checkpoint({"module": loaded}, checkpoint,
        expected_fingerprints={"config": config_fingerprint(snapshot)}, restore_rng=False)
    assert state.optimizer_step == 1
    assert torch.equal(module.weight, loaded.weight)
    snapshot["training"]["learning_rate"] = 0.5
    with pytest.raises(IncompatibleCheckpointError, match="fingerprint mismatch: config"):
        load_training_checkpoint({"module": loaded}, checkpoint,
            expected_fingerprints={"config": config_fingerprint(snapshot)}, restore_rng=False)


def test_wandb_receives_no_plaintext_credentials(tmp_path):
    config = _config() | {"provider": {"api_key": "private-config", "access_token": "private-token"}}
    context = RunContext.create(tmp_path / "run", config=config,
        command=["train", "--api-key=private-command"],
        environment={"HF_TOKEN": "private-environment"})
    captured = {}

    class CapturingRun:
        id = "test"

        def save(self, path, **kwargs):
            captured.setdefault("files", []).append(Path(path).read_text(encoding="utf-8"))

        def log_artifact(self, artifact, **kwargs):
            captured["artifact_metadata"] = artifact.metadata
            captured["artifact_files"] = [Path(entry.local_path).read_text(encoding="utf-8")
                                         for entry in artifact.manifest.entries.values()]

    def initialize(**kwargs):
        captured["config"] = kwargs["config"]
        return CapturingRun()

    tracker = RunTracker.create(context, mode="offline", project="tests", initializer=initialize)
    tracker.log_config()
    tracker.log_manifest(context.run_manifest_path)
    checkpoint = context.root / "checkpoints/probe"
    checkpoint.mkdir()
    (checkpoint / "model.safetensors").write_text("dummy-weights", encoding="utf-8")
    tracker.log_best_weights(checkpoint, optimizer_step=1)
    stored = json.dumps(captured) + "\n".join(p.read_text(encoding="utf-8")
        for p in context.root.rglob("*") if p.is_file())
    for secret in ("private-config", "private-token", "private-command", "private-environment"):
        assert secret not in stored
    assert captured["config"]["data"]["max_context_tokens"] == 8192


def test_credential_free_fingerprint_remains_compatible():
    config = _config()
    historical = hashlib.sha256(canonical_config_json(config).encode("utf-8")).hexdigest()
    assert config_fingerprint(config) == historical


@pytest.mark.parametrize("key", ["image_token", "eot_token", "custom_special_token"])
def test_vocabulary_token_values_are_semantic(key):
    config = _config() | {"tokenizer": {key: "<first>"}}
    assert run_module.redact_secrets(config) == config
    changed = deepcopy(config)
    changed["tokenizer"][key] = "<second>"
    assert config_fingerprint(changed) != config_fingerprint(config)


def test_historical_resume_sanitizes_requested_config_and_wandb_manifests(tmp_path):
    config = _config() | {"provider": {"private_key": "historical-config-secret"}}
    context = RunContext.create(tmp_path / "run", config=config, command=["train"])
    # Simulate artifacts left by the old copy/raw and substring-based writers.
    (context.root / "config/requested.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    (context.root / "config/resolved.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    manifest = json.loads(context.run_manifest_path.read_text())
    manifest["command"] = ["train", "--private-key", "historical-command-secret"]
    context.run_manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    environment = context.root / "manifests/environment.json"
    environment.write_text(json.dumps({"environment": {"PRIVATE_KEY": "historical-env-secret"}}))
    resumed = RunContext.create(context.root, config=config, command=["train"], resume=True)
    captured = []

    class CapturingRun:
        def save(self, path, **kwargs):
            captured.append(Path(path).read_text(encoding="utf-8"))

    tracker = RunTracker.create(resumed, mode="offline", project="tests",
                                initializer=lambda **kwargs: CapturingRun())
    tracker.log_config()
    tracker.log_manifest(context.run_manifest_path)
    stored = "\n".join(captured + [p.read_text(encoding="utf-8") for p in context.root.rglob("*")
                                    if p.is_file()])
    for secret in ("historical-config-secret", "historical-command-secret", "historical-env-secret"):
        assert secret not in stored


def test_historical_fingerprints_require_reconstructed_original_config(tmp_path):
    config = _config() | {"provider": {"api_key": "historical-private-value"},
                          "runtime": {"device": "cpu"}}
    old_normalized = deepcopy(config)
    del old_normalized["runtime"]["device"]
    hashes = {
        hashlib.sha256(canonical_config_json(old_normalized).encode()).hexdigest(),
        hashlib.sha256(canonical_config_json(config).encode()).hexdigest(),
    }
    for index, historical in enumerate(hashes):
        checkpoint = tmp_path / str(index)
        save_training_checkpoint({}, checkpoint, state=TrainerState(config_fingerprint=historical))
        before = (checkpoint / "state.json").read_bytes()
        state = load_training_checkpoint({}, checkpoint, expected_fingerprints={
            "config": config_module.config_fingerprint_candidates(config)}, restore_rng=False)
        assert state.config_fingerprint == historical
        assert (checkpoint / "state.json").read_bytes() == before
        changed = deepcopy(config)
        changed["training"]["learning_rate"] = 0.5
        with pytest.raises(IncompatibleCheckpointError):
            load_training_checkpoint({}, checkpoint, expected_fingerprints={
                "config": config_module.config_fingerprint_candidates(changed)}, restore_rng=False)


def test_training_requested_config_does_not_copy_plaintext_secrets(tmp_path):
    from doc_to_atom.training import cli
    from test_training_loop import _record

    config = load_config(Path(__file__).parents[1] / "configs/experiments/small_overfit.yaml")
    config["provider"] = {"api_key": "private-requested-value"}
    requested = tmp_path / "input.yaml"
    requested.write_text(yaml.safe_dump(config), encoding="utf-8")
    data = tmp_path / "atoms.jsonl"
    data.write_text(_record("train").model_dump_json() + "\n", encoding="utf-8")
    output = tmp_path / "run"
    cli.execute(cli.build_parser().parse_args([
        "--config", str(requested), "--data", str(data), "--output", str(output),
        "--wandb-mode", "disabled", "--validate-only",
    ]))
    for path in output.rglob("*"):
        if path.is_file():
            assert "private-requested-value" not in path.read_text(encoding="utf-8")


@pytest.mark.integration
def test_real_training_cli_checkpoint_evaluation_roundtrip(tmp_path, monkeypatch):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import Gemma2Config, Gemma2ForCausalLM, PreTrainedTokenizerFast
    from doc_to_atom.evaluation import cli as evaluation
    from doc_to_atom.training import cli as training
    from test_training_loop import _record

    # Real local model/tokenizer: no network or mocked training/checkpoint/evaluation.
    model_path = tmp_path / "tiny-gemma"
    Gemma2ForCausalLM(Gemma2Config(vocab_size=32, hidden_size=16, intermediate_size=32,
        num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=8,
        max_position_embeddings=256, pad_token_id=0, bos_token_id=1, eos_token_id=2,
    )).save_pretrained(model_path)
    backend = Tokenizer(models.WordLevel({"[PAD]": 0, "[BOS]": 1, "[EOS]": 2,
        "[UNK]": 3, "Alpha": 4, "What": 5, "?": 6, "Question": 7, "Answer": 8, ":": 9},
        unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", bos_token="[BOS]",
        eos_token="[EOS]", unk_token="[UNK]", model_max_length=256,
        model_input_names=["input_ids", "attention_mask"]).save_pretrained(model_path)
    config = load_config(Path(__file__).parents[1] / "configs/experiments/small_overfit.yaml")
    config["model"].update(name=str(model_path), hidden_size=16, decoder_layers=4,
        encoder_layers=2, memory_layer_indices=[2, 3], flash_attention_2=False)
    config["runtime"].update(device="cpu", checkpoint_upload="none")
    config["training"].update(max_steps=1, max_epochs=1, validation_every_steps=1)
    config["data"].update(max_context_tokens=128, max_atom_tokens=32, max_answer_tokens=16,
                         max_questions_per_sample=1)
    config["evaluation"]["max_new_tokens"] = 2
    config["provider"] = {"api_key": "private-integration-value"}
    config_path = tmp_path / "input.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    train, validation = _record("train"), _record("validation")
    data = tmp_path / "atoms.jsonl"
    data.write_text(train.model_dump_json() + "\n" + validation.model_dump_json() + "\n", encoding="utf-8")
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps({"id": r.record_id, "context": r.context,
        "questions": [{"id": qa.question_id, "question": qa.question, "answers": list(qa.answers)}
                      for qa in r.qa_pairs]}) + "\n" for r in [train, validation]), encoding="utf-8")
    output = tmp_path / "run"
    result = training.execute(training.build_parser().parse_args([
        "--config", str(config_path), "--data", str(data), "--output", str(output),
        "--wandb-mode", "disabled", "--no-reranker",
    ]))
    assert result["training"]["optimizer_steps"] == 1
    snapshot = output / "config/resolved.yaml"
    args = ["doc2atom-evaluate", "--config", str(snapshot), "--data", str(data),
            "--source", str(source), "--checkpoint", str(output / "checkpoints/best.json"),
            "--output", str(output / "predictions.jsonl"), "--no-reranker"]
    monkeypatch.setattr("sys.argv", args)
    evaluation.main()
    predictions = [json.loads(line) for line in (output / "predictions.jsonl").read_text().splitlines()]
    assert len(predictions) == 1 and predictions[0]["checkpoint_step"] == 1
    changed = load_config(snapshot)
    changed["training"]["learning_rate"] *= 2
    snapshot.write_text(yaml.safe_dump(changed), encoding="utf-8")
    args[args.index("--output") + 1] = str(output / "changed-predictions.jsonl")
    with pytest.raises(IncompatibleCheckpointError, match="fingerprint mismatch: config"):
        evaluation.main()
