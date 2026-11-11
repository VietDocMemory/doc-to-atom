"""Local Qwen inference for the shared three-stage XML annotation pipeline.

Model weights may be downloaded from the Hub on first use; annotation never
uses an inference API. Loading is lazy so cache-only resumes need no GPU.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from .client import MiniMaxAPIError, ThreeStageXMLAtomizer
from .models import AnnotationMeta

DEFAULT_LOCAL_QWEN_MODEL = "Qwen/Qwen3-14B"


@dataclass
class LocalQwenConfig:
    model: str = DEFAULT_LOCAL_QWEN_MODEL
    max_completion_tokens: int = 8192
    irrelevant_probe_count: int = 2

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("local model ID or path must not be empty")
        if self.max_completion_tokens < 1 or self.irrelevant_probe_count < 0:
            raise ValueError("completion budget must be positive and probe count nonnegative")


class LocalQwenAtomizer(ThreeStageXMLAtomizer):
    def __init__(self, config: LocalQwenConfig):
        # The shared pipeline only needs model, budget, and probe count.
        super().__init__(config)
        self.provider = "local_qwen"
        self._tokenizer = None
        self._model = None

    def _load_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("local_qwen requires a CUDA GPU; no API fallback is allowed")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model)
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model, torch_dtype=torch.bfloat16, device_map="auto",
        )
        model.eval()
        self._tokenizer, self._model = tokenizer, model

    def _request(self, system: str, user: str) -> str:
        import torch

        # Keep initialization/runtime errors out of the pipeline's per-record
        # XML recovery handlers. MiniMaxAPIError is its historical fatal error.
        try:
            self._load_model()
        except Exception as exc:
            raise MiniMaxAPIError("Local Qwen initialization failed; no API fallback") from exc
        started = time.perf_counter()
        parameters = {
            "max_new_tokens": self.config.max_completion_tokens,
            "do_sample": False,
            "enable_thinking": False,
        }
        try:
            text = self._tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False,
            )
            inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
            prompt_tokens = inputs["input_ids"].shape[-1]
            context_limit = getattr(self._model.config, "max_position_embeddings", None)
            if context_limit and prompt_tokens + self.config.max_completion_tokens > context_limit:
                raise MiniMaxAPIError(
                    "Local Qwen context budget exceeded; input was not truncated"
                )
            with torch.inference_mode():
                generated = self._model.generate(
                    **inputs, max_new_tokens=self.config.max_completion_tokens, do_sample=False,
                )
            output_ids = generated[0][prompt_tokens:].tolist()
            content = self._tokenizer.decode(output_ids, skip_special_tokens=True)
            eos_ids = self._model.generation_config.eos_token_id
            if eos_ids is None:
                eos_ids = self._tokenizer.eos_token_id
            if isinstance(eos_ids, int):
                eos_ids = [eos_ids]
            stopped = bool(output_ids and output_ids[-1] in (eos_ids or []))
            finish_reason = "stop" if stopped else "length"
            self._response_metadata = {
                "raw_body": json.dumps({
                    "content": content, "token_ids": output_ids,
                    "finish_reason": finish_reason, "prompt_tokens": prompt_tokens,
                }, ensure_ascii=False),
                "requested_model": self.config.model,
                "request_parameters": parameters,
                "latency_ms": (time.perf_counter() - started) * 1000,
            }
            if not stopped or not content.strip():
                self._response_error = (
                    f"Local Qwen incomplete completion ({finish_reason}); inspect the sidecar "
                    "before increasing the token budget or explicitly retrying"
                )
            return content
        except MiniMaxAPIError:
            raise
        except Exception as exc:
            raise MiniMaxAPIError("Local Qwen generation failed; no API fallback or retry") from exc

    def _stamp_provenance(self, meta: AnnotationMeta) -> None:
        meta.prompt_provenance = (
            f"provider=local_qwen; requested_model={self.config.model}; "
            f"model={self.resolved_model}; enable_thinking=false"
        )
