# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlexAttention compatibility for DFlash2 padded KV pages.

DFlash2 heterogeneous KV support creates a block-first KV tensor with a
physical page stride (32768-byte virtual block stride). FlexAttention's legacy
path assumes the cache can be flattened with view(-1) before creating the
attention tensors. This module keeps the existing FlexAttention implementation
unchanged for normal contiguous caches and only materializes a contiguous view
for DFlash2 padded KV tensors.
"""

from __future__ import annotations

from functools import wraps

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_INSTALLED = False


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
    def forward_with_dflash2_kv_compat(self, layer, query, key, value,
                                       kv_cache, attn_metadata, output,
                                       output_scale=None,
                                       output_block_scale=None):
        if kv_cache is not None and not kv_cache.is_contiguous():
            # FlexAttention currently reshapes K/V cache with view().
            # DFlash2's padded KV layout is intentionally non-contiguous.
            # Make a local contiguous snapshot only for FlexAttention; the
            # original KV storage and scheduler block table remain unchanged.
            kv_cache = kv_cache.contiguous()
            logger.info_once(
                "DFlash2 FlexAttention compatibility materialized a "
                "contiguous KV view for padded heterogeneous KV cache."
            )
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
