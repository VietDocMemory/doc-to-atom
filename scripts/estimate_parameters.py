from __future__ import annotations

import json


def linear(in_dim: int, out_dim: int) -> int:
    return in_dim * out_dim + out_dim


def estimate_gemma() -> dict:
    hidden, latent, rank, layers = 2304, 256, 8, 4
    shapes = [(2304, 2048), (2304, 1024), (2048, 2304), (9216, 2304)]
    projection = 2 * hidden + linear(hidden, latent) + linear(latent, latent)
    trunk = 2 * latent + linear(latent, latent) + linear(latent, latent)
    key_head = 2 * latent + linear(latent, latent)
    factor_outputs = layers * sum(rank * (input_dim + output_dim) for input_dim, output_dim in shapes)
    factor_heads = linear(latent, factor_outputs)
    mask_head = linear(latent, layers * len(shapes))
    router = 2 * latent + linear(latent, latent) + 4 + 5
    trainable = projection + trunk + key_head + factor_heads + mask_head + router
    return {
        "projection": projection,
        "compiler_trunk": trunk,
        "key_head": key_head,
        "direct_factor_heads_assumption": factor_heads,
        "mask_head": mask_head,
        "router_without_frozen_bge": router,
        "trainable_total_assuming_direct_parallel_heads": trainable,
        "trainable_fp32_mb": trainable * 4 / 1024**2,
        "adamw_params_grads_moments_gb": trainable * 16 / 1024**3,
        "warning": "Compiler head factorization is PAPER_UNSPECIFIED; official trainable count may differ.",
    }


if __name__ == "__main__":
    print(json.dumps(estimate_gemma(), indent=2))

