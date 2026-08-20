# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlexAttention compatibility for DFlash2 padded KV pages.

DFlash2 heterogeneous KV support creates a block-first KV tensor with a
physical page stride (32768-byte virtual block stride). FlexAttention's legacy
path assumes the cache geometry is directly usable by its sparse block kernel.
This module keeps the existing FlexAttention implementation unchanged for
normal caches and normalizes only the metadata/cache view boundary for DFlash2.
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
    """Make sparse FlexAttention block geometry Triton compatible.

    The DFlash2 logical KV page can be split into 135 virtual kernel blocks.
    The cache tensor itself must keep that layout, but FlexAttention's sparse
    Triton kernels require compile-time arange extents to be powers of two.
    Use a power-of-two KV block and let the existing block mask handle the
    padded tail.
    """
    kv_block_size = getattr(attn_metadata, "kv_block_size", None)
    if kv_block_size is None or _is_power_of_two(kv_block_size):
        return

    # 2160-token DFlash logical blocks are represented by the padded cache
    # compatibility layer as 16-token virtual blocks. Round any remaining
    # non-power-of-two metadata value upward; masking handles invalid tokens.
    padded = 1 << (kv_block_size - 1).bit_length()
    if padded != kv_block_size:
        attn_metadata.kv_block_size = padded
        # Force FlexAttention to rebuild BlockMask using the legal geometry.
        attn_metadata.block_mask = None
        logger.info_once(
            "DFlash2 FlexAttention compatibility changed KV block geometry "
            "from %d to %d for Triton power-of-two sparse kernels.",
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
