# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""FlexAttention compatibility for DFlash2 padded KV pages.

DFlash2 heterogeneous KV support keeps a 2160-token logical cache block while
FlexAttention's direct sparse BlockMask path normally reuses that logical block
size as the Triton ``BLOCK_N`` tile. Triton requires ``tl.arange`` extents to be
powers of two, so ``BLOCK_N=2160`` cannot compile.

This compatibility layer keeps the cache tensor, block table, logical
``block_size`` and FlexAttention ``kv_block_size`` unchanged. It only routes an
illegal direct-build kernel tile through FlexAttention's existing non-direct
kernel-option selector, which chooses a small power-of-two divisor of the
logical mask block (16 for the observed 2160-token SM75 path).
"""

from __future__ import annotations

from functools import wraps
from typing import Any, Callable

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_INSTALLED = False


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def _select_legal_direct_kernel_options(
    original: Callable[..., dict[str, int | bool]],
    query: torch.Tensor,
    block_m: int,
    block_n: int,
    use_direct_build: bool,
) -> dict[str, int | bool]:
    """Select legal Triton tiles without changing logical BlockMask geometry.

    FlexAttention's direct-build branch returns ``BLOCK_M/N`` exactly as stored
    in ``BlockMask.BLOCK_SIZE``. That is fine for the usual power-of-two cache
    blocks but fails for DFlash2's 2160-token logical block. The backend already
    has a non-direct option-selection path that chooses smaller divisors for the
    compute kernel. Reuse that selector while leaving the direct BlockMask and
    all metadata intact.
    """
    if not use_direct_build or (
        _is_power_of_two(block_m) and _is_power_of_two(block_n)
    ):
        return original(query, block_m, block_n, use_direct_build)

    options = original(query, block_m, block_n, False)
    tile_m = int(options.get("BLOCK_M", 0))
    tile_n = int(options.get("BLOCK_N", 0))

    if not _is_power_of_two(tile_m) or not _is_power_of_two(tile_n):
        raise RuntimeError(
            "DFlash2 FlexAttention kernel-option fallback did not produce "
            f"power-of-two Triton tiles: BLOCK_M={tile_m}, BLOCK_N={tile_n}."
        )
    if block_m % tile_m != 0 or block_n % tile_n != 0:
        raise RuntimeError(
            "DFlash2 FlexAttention kernel tiles must divide the logical "
            f"BlockMask geometry: mask=({block_m}, {block_n}), "
            f"tiles=({tile_m}, {tile_n})."
        )

    logger.info_once(
        "DFlash2 FlexAttention compatibility kept logical BlockMask geometry "
        "(%d, %d) and selected Triton tiles BLOCK_M=%d, BLOCK_N=%d.",
        block_m,
        block_n,
        tile_m,
        tile_n,
    )
    return options


def install_dflash2_flex_attention_compat() -> None:
    global _INSTALLED
    if _INSTALLED:
        return

    from vllm.v1.attention.backends import flex_attention as flex_mod

    # Patch only kernel-option selection. BlockMask metadata remains authoritative
    # and keeps kv_block_size == cache block_size == 2160 on the reported path.
    original_get_kernel_options = flex_mod.get_kernel_options
    if not getattr(
        original_get_kernel_options, "_dflash2_power_of_two_tile_compat", False
    ):

        @wraps(original_get_kernel_options)
        def get_kernel_options_with_dflash2_tiles(
            query: torch.Tensor,
            block_m: int,
            block_n: int,
            use_direct_build: bool,
        ) -> dict[str, int | bool]:
            return _select_legal_direct_kernel_options(
                original_get_kernel_options,
                query,
                block_m,
                block_n,
                use_direct_build,
            )

        get_kernel_options_with_dflash2_tiles._dflash2_power_of_two_tile_compat = (  # type: ignore[attr-defined]
            True
        )
        flex_mod.get_kernel_options = get_kernel_options_with_dflash2_tiles

    FlexAttentionImpl = flex_mod.FlexAttentionImpl
    original_forward = FlexAttentionImpl.forward
    if not getattr(original_forward, "_dflash2_flex_kv_compat", False):

        @wraps(original_forward)
        def forward_with_dflash2_kv_compat(
            self: Any,
            layer: Any,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            kv_cache: torch.Tensor,
            attn_metadata: Any,
            output: torch.Tensor,
            output_scale: torch.Tensor | None = None,
            output_block_scale: torch.Tensor | None = None,
        ) -> torch.Tensor:
            if kv_cache is not None and not kv_cache.is_contiguous():
                # The legacy FlexAttention forward flattens K/V with view().
                # Materialize only the local view consumed by FlexAttention;
                # scheduler/block-table/backing KV storage are unchanged.
                kv_cache = kv_cache.contiguous()
                logger.info_once(
                    "DFlash2 FlexAttention compatibility materialized a "
                    "contiguous KV view for padded heterogeneous KV cache."
                )

            return original_forward(
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

        forward_with_dflash2_kv_compat._dflash2_flex_kv_compat = True  # type: ignore[attr-defined]
        FlexAttentionImpl.forward = forward_with_dflash2_kv_compat

    _INSTALLED = True
