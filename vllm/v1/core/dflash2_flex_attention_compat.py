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


def _remap_block_indices(
    indices: torch.Tensor | None, remap: torch.Tensor
) -> torch.Tensor | None:
    if indices is None:
        return None
    valid = indices >= 0
    safe_indices = indices.clamp_min(0).long()
    mapped = remap[safe_indices].to(indices.dtype)
    return torch.where(valid, mapped, torch.full_like(indices, -1))


def _compact_single_request_kv(
    kv_cache: torch.Tensor, attn_metadata: Any
) -> tuple[torch.Tensor, Callable[[], None]] | None:
    """Pack only the active physical blocks for the batch=1 DFlash2 path.

    The old compatibility path materializes the entire padded KV pool.  For a
    single request, the block table already identifies the small subset needed
    by FlexAttention.  We temporarily remap its BlockMask and physical-to-
    logical mapping to a compact block numbering, then restore metadata after
    the forward call.
    """
    if kv_cache.is_contiguous() or getattr(attn_metadata, "num_reqs", 0) != 1:
        return None
    block_mask = getattr(attn_metadata, "block_mask", None)
    doc_ids = getattr(attn_metadata, "doc_ids", None)
    if block_mask is None or doc_ids is None:
        return None

    # The compact path is explicitly batch=1; avoid a GPU->CPU sync here.
    req = 0
    block_size = int(attn_metadata.block_size)
    num_active = (int(attn_metadata.max_seq_len) + block_size - 1) // block_size
    if num_active <= 0:
        return None

    key_cache, value_cache = kv_cache.unbind(0)
    block_ids = attn_metadata.block_table[req, :num_active].to(torch.long)
    if block_ids.numel() != num_active:
        return None

    compact_key = key_cache.index_select(0, block_ids)
    compact_value = value_cache.index_select(0, block_ids)
    compact_kv = torch.stack((compact_key, compact_value), dim=0)

    remap = torch.full(
        (key_cache.shape[0],), -1, dtype=torch.long, device=block_ids.device
    )
    remap[block_ids] = torch.arange(num_active, device=block_ids.device)

    old_block_table = attn_metadata.block_table
    old_physical_to_logical = attn_metadata.physical_to_logical
    old_total_cache_tokens = attn_metadata.total_cache_tokens
    old_num_blocks = attn_metadata.num_blocks
    old_seq_lengths = block_mask.seq_lengths
    old_kv_indices = block_mask.kv_indices
    old_full_kv_indices = block_mask.full_kv_indices

    compact_block_table = torch.full_like(old_block_table, -1)
    compact_block_table[req, :num_active] = torch.arange(
        num_active, device=old_block_table.device, dtype=old_block_table.dtype
    )
    compact_physical_to_logical = torch.full_like(old_physical_to_logical, -1)
    compact_physical_to_logical[req, :num_active] = old_physical_to_logical[
        req, block_ids
    ]

    attn_metadata.block_table = compact_block_table
    attn_metadata.physical_to_logical = compact_physical_to_logical
    attn_metadata.total_cache_tokens = num_active * block_size
    attn_metadata.num_blocks = num_active
    block_mask.kv_indices = _remap_block_indices(old_kv_indices, remap)
    block_mask.full_kv_indices = _remap_block_indices(old_full_kv_indices, remap)
    block_mask.seq_lengths = (old_seq_lengths[0], num_active * block_size)

    def restore() -> None:
        attn_metadata.block_table = old_block_table
        attn_metadata.physical_to_logical = old_physical_to_logical
        attn_metadata.total_cache_tokens = old_total_cache_tokens
        attn_metadata.num_blocks = old_num_blocks
        block_mask.kv_indices = old_kv_indices
        block_mask.full_kv_indices = old_full_kv_indices
        block_mask.seq_lengths = old_seq_lengths

    logger.info_once(
        "DFlash2 FlexAttention using active-block compact KV for batch=1 "
        "(%d/%d blocks, %.1f MiB logical cache).",
        num_active,
        key_cache.shape[0],
        compact_kv.numel() * compact_kv.element_size() / 2**20,
    )
    return compact_kv, restore


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
            restore_metadata: Callable[[], None] | None = None
            if kv_cache is not None and not kv_cache.is_contiguous():
                # Profiling/KV initialization calls this hook without request
                # metadata; keep the one required contiguous snapshot there.
                if attn_metadata is None:
                    kv_cache = kv_cache.contiguous()
                    logger.info_once(
                        "DFlash2 FlexAttention compatibility materialized a "
                        "contiguous KV view for padded heterogeneous KV cache."
                    )
                else:
                    # Real requests must use compact active-block KV. Never
                    # silently materialize the full padded pool on a metadata
                    # miss or an unsupported batch shape.
                    if getattr(attn_metadata, "num_reqs", 0) > 1:
                        raise RuntimeError(
                            "DFlash2 padded heterogeneous KV currently supports "
                            "batch=1 only; set max_num_seqs=1. Multi-request KV "
                            "compaction is not implemented, so refusing the "
                            "unsafe full-pool contiguous fallback."
                        )
                    if getattr(attn_metadata, "block_mask", None) is None:
                        # Metadata is normally pre-built by the builder, but the
                        # first eager request can arrive before that lazy build.
                        if attn_metadata.direct_build:
                            attn_metadata.block_mask = attn_metadata._build_block_mask_direct()
                        else:
                            attn_metadata.block_mask = attn_metadata.build_block_mask()
                    compact = _compact_single_request_kv(kv_cache, attn_metadata)
                    if compact is None:
                        raise RuntimeError(
                            "DFlash2 compact KV unavailable; refusing unsafe "
                            "full-pool snapshot."
                        )
                    kv_cache, restore_metadata = compact

            try:
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
            finally:
                if restore_metadata is not None:
                    restore_metadata()

        forward_with_dflash2_kv_compat._dflash2_flex_kv_compat = True  # type: ignore[attr-defined]
        FlexAttentionImpl.forward = forward_with_dflash2_kv_compat

    _INSTALLED = True
