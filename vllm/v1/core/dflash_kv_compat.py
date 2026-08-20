# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2 compatibility for heterogeneous KV-cache page sizes.

The vLLM 0.21-era hybrid KV manager assumes a single physical page size
across cache groups. DFlash2 can violate that assumption when a quantized
target cache (for example TurboQuant K8V4) is paired with the draft model's
native FP16 non-causal attention cache.

This module is imported only through the fork-local DFlash2 model shim. It
keeps the legacy scheduler/block-table model unchanged and falls back to
physical page-stride padding when the normal integer-ratio page unifier cannot
represent the mixed cache geometry.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import replace
from functools import wraps

import torch

from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    TQFullAttentionSpec,
)

logger = init_logger(__name__)

_INSTALLED = False
_TQ_STRIDED_STORE_INSTALLED = False


def _pad_nondivisible_page_sizes(
    kv_cache_spec: dict[str, KVCacheSpec],
) -> dict[str, KVCacheSpec]:
    """Pad physical page stride to the largest page without changing block_size.

    ``page_size_padded`` is already honored by both AttentionSpec and MambaSpec
    in this fork. Keeping the logical ``block_size`` unchanged is important for
    DFlash because target and draft share the scheduler's block table.
    """
    page_sizes = {spec.page_size_bytes for spec in kv_cache_spec.values()}
    if len(page_sizes) <= 1:
        return kv_cache_spec

    max_page_size = max(page_sizes)
    padded: dict[str, KVCacheSpec] = {}
    for layer_name, spec in kv_cache_spec.items():
        if spec.page_size_bytes == max_page_size:
            padded[layer_name] = spec
            continue

        if not hasattr(spec, "page_size_padded"):
            raise NotImplementedError(
                "DFlash2 heterogeneous KV-page compatibility requires "
                f"page_size_padded support, but {layer_name} uses "
                f"{type(spec).__name__}."
            )

        new_spec = replace(spec, page_size_padded=max_page_size)
        if new_spec.block_size != spec.block_size:
            raise AssertionError("physical page padding must not change block_size")
        if new_spec.page_size_bytes != max_page_size:
            raise AssertionError(
                f"failed to pad {layer_name} to {max_page_size} bytes"
            )
        padded[layer_name] = new_spec

    return padded


def _install_turboquant_strided_store() -> None:
    """Allow TurboQuant store kernels to receive a page-strided cache view.

    The kernels already compute addresses from ``stride_cache_*``. The legacy
    launcher flattened the tensor with ``view(-1)``, which rejects a
    non-contiguous view whose block stride includes physical page padding.
    For contiguous caches we preserve the original launcher exactly.
    """
    global _TQ_STRIDED_STORE_INSTALLED
    if _TQ_STRIDED_STORE_INSTALLED:
        return

    try:
        store_mod = importlib.import_module(
            "vllm.v1.attention.ops.triton_turboquant_store"
        )
    except ImportError:
        # TurboQuant is optional. If it is not part of this model, there is
        # nothing to patch.
        return

    original = store_mod.triton_turboquant_store
    if getattr(original, "_dflash2_strided_kv_compat", False):
        _TQ_STRIDED_STORE_INSTALLED = True
        return

    @wraps(original)
    def strided_store(
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        PiT: torch.Tensor,
        midpoints: torch.Tensor,
        mse_bits: int,
        key_packed_size: int,
        value_quant_bits: int,
        key_fp8: bool = False,
    ):
        if kv_cache.is_contiguous():
            return original(
                key,
                value,
                kv_cache,
                slot_mapping,
                PiT,
                midpoints,
                mse_bits,
                key_packed_size,
                value_quant_bits,
                key_fp8=key_fp8,
            )

        # This is the legacy launcher with one intentional semantic change:
        # pass the strided cache tensor directly to Triton instead of calling
        # ``view(-1)``. Triton receives the tensor's data pointer while address
        # calculation continues to use the explicit strides below.
        N, H, D = key.shape
        NH = N * H
        block_size = kv_cache.shape[1]
        BLOCK_D = store_mod.triton.next_power_of_2(D)
        mse_bytes = store_mod.math.ceil(D * mse_bits / 8)
        n_centroids = 2**mse_bits
        val_data_bytes = store_mod.math.ceil(D * value_quant_bits / 8)
        BLOCK_VAL = store_mod.triton.next_power_of_2(val_data_bytes)

        # uint8 cache: element strides are byte strides.
        stride_block = kv_cache.stride(0)
        stride_pos = kv_cache.stride(1)
        stride_head = kv_cache.stride(2)
        block_grp = store_mod.triton.next_power_of_2(D // 8) if D >= 8 else 1

        if key_fp8:
            k_flat = key.reshape(NH, D).contiguous()
            v_flat = value.reshape(NH, D).contiguous()
            fp8_format = store_mod._fp8_format_code(key.device.index or 0)

            grid = (NH,)
            store_mod._tq_fused_store_fp8[grid](
                k_flat,
                v_flat,
                kv_cache,
                slot_mapping,
                stride_cache_block=stride_block,
                stride_cache_pos=stride_pos,
                stride_cache_head=stride_head,
                D=D,
                H=H,
                BLOCK_SIZE=block_size,
                BLOCK_D=BLOCK_D,
                KPS=key_packed_size,
                VQB=value_quant_bits,
                VAL_DATA_BYTES=val_data_bytes,
                BLOCK_VAL=BLOCK_VAL,
                BLOCK_GRP=block_grp,
                FP8_FORMAT=fp8_format,
                num_warps=4,
                num_stages=1,
            )
            return None

        k_flat = key.float().reshape(NH, D)
        norms = k_flat.norm(dim=1, keepdim=True)
        x_hat = k_flat / (norms + 1e-8)
        y = x_hat @ PiT
        v_flat = value.float().reshape(NH, D)

        grid = (NH,)
        store_mod._tq_fused_store_mse[grid](
            y,
            norms.squeeze(1),
            v_flat,
            midpoints,
            kv_cache,
            slot_mapping,
            stride_cache_block=stride_block,
            stride_cache_pos=stride_pos,
            stride_cache_head=stride_head,
            D=D,
            H=H,
            BLOCK_SIZE=block_size,
            BLOCK_D=BLOCK_D,
            MSE_BYTES=mse_bytes,
            KPS=key_packed_size,
            VQB=value_quant_bits,
            VAL_DATA_BYTES=val_data_bytes,
            BLOCK_VAL=BLOCK_VAL,
            MSE_BITS=mse_bits,
            N_CENTROIDS=n_centroids,
            BLOCK_GRP=block_grp,
            num_warps=4,
            num_stages=1,
        )
        return None

    strided_store._dflash2_strided_kv_compat = True  # type: ignore[attr-defined]
    store_mod.triton_turboquant_store = strided_store

    # turboquant_attn imports the launcher by name, so update its module-global
    # reference too when that backend has already been imported.
    backend_mod = sys.modules.get("vllm.v1.attention.backends.turboquant_attn")
    if backend_mod is not None and getattr(
        backend_mod, "triton_turboquant_store", None
    ) is original:
        backend_mod.triton_turboquant_store = strided_store

    _TQ_STRIDED_STORE_INSTALLED = True
    logger.info_once(
        "DFlash2 heterogeneous KV compatibility enabled TurboQuant "
        "page-strided cache stores."
    )


def install_dflash2_heterogeneous_kv_compat() -> None:
    """Install the narrow legacy-manager fallback needed by DFlash2.

    The original unifier remains the first choice. We only intervene when it
    raises on non-divisible page sizes, so divisible legacy layouts retain their
    existing block-size transformation and all non-DFlash2 model loading remains
    untouched because this installer is reached only by the DFlash2 model shim.
    """
    global _INSTALLED
    if _INSTALLED:
        return

    from vllm.v1.core import kv_cache_utils

    original = kv_cache_utils.unify_kv_cache_spec_page_size
    if getattr(original, "_dflash2_heterogeneous_kv_compat", False):
        _INSTALLED = True
        return

    @wraps(original)
    def dflash2_unify(
        kv_cache_spec: dict[str, KVCacheSpec],
    ) -> dict[str, KVCacheSpec]:
        try:
            return original(kv_cache_spec)
        except NotImplementedError:
            page_sizes = {spec.page_size_bytes for spec in kv_cache_spec.values()}
            if len(page_sizes) <= 1:
                raise

            max_page_size = max(page_sizes)
            # If every page divides the largest page, the original helper should
            # have succeeded; do not mask an unrelated planner failure.
            if all(max_page_size % page_size == 0 for page_size in page_sizes):
                raise

            # This backport is intentionally narrow: only the observed
            # TurboQuant-target + native-FP16 DFlash cache combination is
            # enabled. Do not silently broaden arbitrary hybrid models.
            has_turboquant = any(
                isinstance(spec, TQFullAttentionSpec)
                for spec in kv_cache_spec.values()
            )
            has_native_full_attention = any(
                isinstance(spec, FullAttentionSpec)
                and not isinstance(spec, TQFullAttentionSpec)
                for spec in kv_cache_spec.values()
            )
            if not (has_turboquant and has_native_full_attention):
                raise

            # The existing padded-view implementation is safe for the
            # TurboQuant block-first cache and the SM75 Triton DFlash cache.
            # Reject any additional smaller standard-attention format rather
            # than assuming its backend has a blocks-first physical layout.
            for spec in kv_cache_spec.values():
                if (
                    isinstance(spec, AttentionSpec)
                    and spec.page_size_bytes < max_page_size
                    and not isinstance(spec, TQFullAttentionSpec)
                ):
                    raise NotImplementedError(
                        "DFlash2 heterogeneous KV-page compatibility only "
                        "pads smaller TurboQuant attention pages; encountered "
                        f"{type(spec).__name__} with page size "
                        f"{spec.page_size_bytes}."
                    )

            padded = _pad_nondivisible_page_sizes(kv_cache_spec)
            _install_turboquant_strided_store()
            logger.info_once(
                "DFlash2 is using heterogeneous KV-cache pages with a shared "
                "logical block table. Physical page strides are padded to "
                "%d bytes while per-layer cache formats and logical block_size "
                "remain unchanged (original page sizes: %s).",
                max_page_size,
                sorted(page_sizes),
            )
            return padded

    dflash2_unify._dflash2_heterogeneous_kv_compat = True  # type: ignore[attr-defined]
    kv_cache_utils.unify_kv_cache_spec_page_size = dflash2_unify
    _INSTALLED = True
