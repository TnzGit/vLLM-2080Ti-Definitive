# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlexAttention compatibility for DFlash2 padded KV pages.

DFlash2 heterogeneous KV support creates a block-first KV tensor with a
physical page stride. FlexAttention's sparse path has stricter compile-time
block geometry requirements than the cache layout itself. Keep cache storage
unchanged and normalize only the metadata used by the sparse kernel.
"""

from __future__ import annotations

from functools import wraps

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_INSTALLED = False


def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _fix_flex_block_geometry(attn_metadata) -> None:
    """Normalize sparse-kernel geometry without creating huge Triton kernels.

    The previous workaround rounded 2160 directly to 4096. That avoids the
    invalid arange extent but can make the first sparse kernel compilation
    excessively large. FlexAttention only needs the kernel tile geometry to be
    legal; the mask still describes the real KV length.

    Use the smallest legal power-of-two tile that covers the virtual KV block.
    The cache tensor and scheduler block table are intentionally untouched.
    """
    kv_block_size = getattr(attn_metadata, "kv_block_size", None)
    if kv_block_size is None or _is_power_of_two(kv_block_size):
        return

    # DFlash2's 2160-token logical page is represented by 135 virtual blocks.
    # A 2048 tile keeps Triton happy while avoiding the large 4096 sparse
    # compilation path. The block mask rebuild pads only the final tail.
    if kv_block_size == 2160:
        padded = 2048
    else:
        padded = 1 << (kv_block_size - 1).bit_length()

    attn_metadata.kv_block_size = padded
    attn_metadata.block_mask = None
    logger.info_once(
        "DFlash2 FlexAttention compatibility changed KV block geometry "
        "from %d to %d for Triton sparse kernels.",
        kv_block_size,
        padded,
    )


def install_dflash2_flex_attention_compat() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from vllm.v1.attention.backends.flex_attention import FlexAttentionImpl

    original = FlexAttentionImpl.forward
    if getattr(original, "_dflash2_flex_kv_compat", False):
        _INSTALLED = True
        return

    @wraps(original)
    def forward_with_dflash2_kv_compat(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        if kv_cache is not None and not kv_cache.is_contiguous():
            kv_cache = kv_cache.contiguous()
            logger.info_once(
                "DFlash2 FlexAttention compatibility materialized a "
                "contiguous KV view for padded heterogeneous KV cache."
            )

        if attn_metadata is not None:
            _fix_flex_block_geometry(attn_metadata)

        return original(
            self,
            layer,
            query,
            key,
            value,
            kv_cache,
            attn_metadata,
            output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )

    forward_with_dflash2_kv_compat._dflash2_flex_kv_compat = True
    FlexAttentionImpl.forward = forward_with_dflash2_kv_compat
    _INSTALLED = True
