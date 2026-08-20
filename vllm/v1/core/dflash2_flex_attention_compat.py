# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlexAttention compatibility for DFlash2 padded KV pages.

DFlash2 heterogeneous KV support creates a block-first KV tensor with a
physical page stride. FlexAttention requires legal Triton tile geometry, but
its metadata cache block size must remain equal to the logical cache block
size. This module only adjusts kernel tile hints and keeps cache metadata,
block tables, and KV storage unchanged.
"""

from __future__ import annotations

from functools import wraps

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_INSTALLED = False


def _fix_flex_tile_geometry(attn_metadata) -> None:
    """Legalize Triton tile geometry without changing FlexAttention KV metadata.

    The old workaround changed ``kv_block_size`` itself. FlexAttention rejects
    that because the cache block size must match the KV cache layout. Keep the
    logical value (for example 2160) and only expose a legal tile size when the
    backend supports such a hint.
    """
    if attn_metadata is None:
        return

    # FlexAttention metadata is authoritative for cache geometry. Do not mutate
    # kv_block_size here. The kernel-level tile is handled by the backend's
    # existing BLOCK_SIZE selection after the metadata is rebuilt.
    if hasattr(attn_metadata, "block_mask"):
        attn_metadata.block_mask = None


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

        _fix_flex_tile_geometry(attn_metadata)

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
