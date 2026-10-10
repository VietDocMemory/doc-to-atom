from __future__ import annotations

import logging

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from doc_to_atom.compiler.model import AtomMemoryCompiler
from doc_to_atom.composer.model import MemoryComposer
from doc_to_atom.encoder.shared import SharedAtomQueryEncoder
from doc_to_atom.injection.lora import (
    DynamicLoRAInjector,
    discover_target_shapes,
    freeze_base_model,
)
from doc_to_atom.router.model import TwoStageRouter
from doc_to_atom.router.reranker import FrozenBGEReranker
from doc_to_atom.training.system import TeacherStudentSystem

LOGGER = logging.getLogger(__name__)


def build_system(config: dict, enable_reranker: bool = True):
    model_cfg = config["model"]
    dtype = torch.bfloat16
    kwargs = {"torch_dtype": dtype}
    if model_cfg.get("flash_attention_2") is True:
        kwargs["attn_implementation"] = "flash_attention_2"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_cfg["name"], **kwargs)
    except (ImportError, ValueError) as exc:
        if config.get("runtime", {}).get("flash_attention_2") == "auto":
            LOGGER.warning(
                "REPRODUCTION DEVIATION: FlashAttention-2 unavailable; using SDPA: %s", exc
            )
            kwargs["attn_implementation"] = "sdpa"
            model = AutoModelForCausalLM.from_pretrained(model_cfg["name"], **kwargs)
        else:
            raise
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["name"])
    freeze_base_model(model)
    targets = discover_target_shapes(
        model, model_cfg["memory_layer_indices"], model_cfg["target_modules"]
    )
    encoder = SharedAtomQueryEncoder(
        model,
        model_cfg["hidden_size"],
        config["encoder"]["output_dim"],
        model_cfg["encoder_layers"],
    )
    compiler_cfg = config["compiler"]
    micro_kv = compiler_cfg["micro_kv"]
    if micro_kv["enabled"]:
        raise NotImplementedError(
            "micro-KV is an optional paper interface, but its shape, positional semantics, and "
            "injection procedure are PAPER_UNSPECIFIED; refusing to invent a main-setting implementation"
        )
    compiler = AtomMemoryCompiler(
        targets,
        input_dim=config["encoder"]["output_dim"],
        latent_dim=compiler_cfg["latent_dim"],
        key_dim=compiler_cfg["key_dim"],
        rank=compiler_cfg["lora_rank"],
        a_std=compiler_cfg["lora_a_std"],
        micro_kv_slots=micro_kv["slots"] if micro_kv["enabled"] else None,
        micro_kv_head_dim=micro_kv["head_dim"] if micro_kv["enabled"] else None,
    )
    reranker = (
        FrozenBGEReranker(
            config["router"]["reranker_name"], config["router"]["reranker_max_length"]
        )
        if enable_reranker
        else None
    )
    router = TwoStageRouter(
        stage1_top_k=config["router"]["stage1_top_k"],
        rerank_fusion_weight=config["router"]["rerank_fusion_weight"],
        temperature=config["router"]["temperature"],
        reranker=reranker,
    )
    composer = MemoryComposer()
    injector = DynamicLoRAInjector(
        model, targets, compiler_cfg["lora_rank"], compiler_cfg["lora_alpha"]
    )
    system = TeacherStudentSystem(model, encoder, compiler, router, composer, injector)
    return system, tokenizer, targets


def parameter_counts(module: torch.nn.Module) -> dict[str, int]:
    return {
        "frozen": sum(
            parameter.numel() for parameter in module.parameters() if not parameter.requires_grad
        ),
        "trainable": sum(
            parameter.numel() for parameter in module.parameters() if parameter.requires_grad
        ),
    }
