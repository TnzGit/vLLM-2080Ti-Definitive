# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Worker-side DFlash2 compatibility for padded TurboQuant KV pages.

The legacy GPUModelRunner reshapes attention cache allocations into the kernel
block geometry. When a logical KV-manager block is split into multiple kernel
blocks, a padded logical page must be split across those kernel blocks as well.
Otherwise the full logical-page stride is incorrectly applied to every virtual
kernel block and ``torch.as_strided`` addresses far beyond the allocation.

This module keeps the fix DFlash2-only by patching the runner when the DFlash2
model shim is loaded in a worker process. The underlying allocation remains the
one produced by EngineCore: ``num_logical_blocks * padded_logical_page_bytes``.
"""

from __future__ import annotations

import importlib
import threading
from dataclasses import dataclass
from functools import wraps
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.utils.torch_utils import get_dtype_size
from vllm.v1.kv_cache_interface import (
    TQFullAttentionSpec,
    UniformTypeKVCacheSpecs,
)

logger = init_logger(__name__)

_TQ_PADDED_RESHAPE_INSTALLED = False
_RESHAPE_PATCH_LOCK = threading.RLock()


@dataclass(frozen=True)
class _TQStridePlan:
    physical_shape: tuple[int, ...]
    physical_block_dim: int
    buggy_stride_elements: int
    kernel_stride_elements: int
    logical_page_bytes: int
    kernel_stride_bytes: int
    virtual_block_ratio: int


def _tq_kernel_page_stride_bytes(
    spec: TQFullAttentionSpec,
    kernel_block_size: int,
) -> int:
    """Return padded bytes between adjacent virtual TQ kernel blocks."""
    if spec.page_size_padded is None:
        raise ValueError("TQ kernel-page stride helper requires a padded page")
    if kernel_block_size <= 0 or spec.block_size % kernel_block_size != 0:
        raise NotImplementedError(
            "DFlash2 padded TurboQuant pages require kernel_block_size to "
            f"divide logical block_size ({kernel_block_size} vs {spec.block_size})."
        )

    ratio = spec.block_size // kernel_block_size
    logical_page_bytes = spec.page_size_bytes
    real_page_bytes = spec.real_page_size_bytes
    if logical_page_bytes % ratio != 0 or real_page_bytes % ratio != 0:
        raise NotImplementedError(
            "DFlash2 padded TurboQuant page cannot be represented with a "
            "uniform virtual-block stride: logical/real page sizes must both "
            f"be divisible by split ratio {ratio} (got {logical_page_bytes} "
            f"and {real_page_bytes} bytes)."
        )

    kernel_stride_bytes = logical_page_bytes // ratio
    kernel_payload_bytes = real_page_bytes // ratio
    if kernel_stride_bytes < kernel_payload_bytes:
        raise AssertionError("padded kernel stride cannot be smaller than payload")
    return kernel_stride_bytes


def _build_tq_stride_plans(
    model_runner: Any,
    kv_cache_raw_tensors: dict[str, torch.Tensor],
    kernel_block_sizes: list[int],
) -> dict[tuple[int, tuple[int, ...]], _TQStridePlan]:
    """Build exact as_strided replacements for padded block-first TQ caches."""
    plans: dict[tuple[int, tuple[int, ...]], _TQStridePlan] = {}

    for group in model_runner._kv_cache_spec_attn_group_iterator():
        group_id = group.kv_cache_group_id
        if group_id >= len(kernel_block_sizes):
            continue
        kernel_block_size = kernel_block_sizes[group_id]
        group_spec = group.kv_cache_spec
        backend = group.backend

        for layer_name in group.layer_names:
            if layer_name not in kv_cache_raw_tensors:
                continue
            spec = group_spec
            if isinstance(spec, UniformTypeKVCacheSpecs):
                spec = spec.kv_cache_specs[layer_name]
            if not isinstance(spec, TQFullAttentionSpec):
                continue
            if spec.page_size_padded is None:
                continue
            if spec.block_size == kernel_block_size:
                # No virtual splitting: the legacy full-page stride is correct.
                continue

            raw_tensor = kv_cache_raw_tensors[layer_name]
            raw_bytes = raw_tensor.numel() * raw_tensor.element_size()
            logical_page_bytes = spec.page_size_bytes
            if raw_bytes % logical_page_bytes != 0:
                raise AssertionError(
                    f"TurboQuant raw allocation for {layer_name} ({raw_bytes} bytes) "
                    f"is not divisible by padded page size {logical_page_bytes}."
                )
            num_logical_blocks = raw_bytes // logical_page_bytes
            ratio = spec.block_size // kernel_block_size
            kernel_num_blocks = num_logical_blocks * ratio

            shape_block_size = (
                spec.storage_block_size
                if spec.storage_block_size != spec.block_size
                else kernel_block_size
            )
            logical_shape = backend.get_kv_cache_shape(
                kernel_num_blocks,
                shape_block_size,
                spec.num_kv_heads,
                spec.head_size,
                cache_dtype_str=model_runner.cache_config.cache_dtype,
            )
            block_dim = backend.get_kv_cache_block_dim(
                kernel_block_size,
                spec.num_kv_heads,
                spec.head_size,
                cache_dtype_str=model_runner.cache_config.cache_dtype,
            )
            if block_dim != 0:
                raise NotImplementedError(
                    "DFlash2 padded TurboQuant compatibility requires a "
                    f"block-first cache backend; got block dimension {block_dim}."
                )

            try:
                stride_order = backend.get_kv_cache_stride_order()
                if len(stride_order) != len(logical_shape):
                    raise AssertionError("invalid KV-cache stride order")
            except (AttributeError, NotImplementedError):
                stride_order = tuple(range(len(logical_shape)))

            physical_shape = tuple(logical_shape[i] for i in stride_order)
            physical_block_dim = stride_order.index(block_dim)
            dtype_size = get_dtype_size(spec.dtype)
            kernel_stride_bytes = _tq_kernel_page_stride_bytes(spec, kernel_block_size)
            if (
                logical_page_bytes % dtype_size != 0
                or kernel_stride_bytes % dtype_size != 0
            ):
                raise NotImplementedError(
                    "DFlash2 padded TurboQuant page stride must align to cache dtype."
                )

            plan = _TQStridePlan(
                physical_shape=physical_shape,
                physical_block_dim=physical_block_dim,
                buggy_stride_elements=logical_page_bytes // dtype_size,
                kernel_stride_elements=kernel_stride_bytes // dtype_size,
                logical_page_bytes=logical_page_bytes,
                kernel_stride_bytes=kernel_stride_bytes,
                virtual_block_ratio=ratio,
            )
            key = (raw_tensor.data_ptr(), physical_shape)
            existing = plans.get(key)
            if existing is not None and existing != plan:
                raise AssertionError("conflicting TurboQuant padded-stride plans")
            plans[key] = plan

    return plans


class _TorchAsStridedProxy:
    """Delegate torch APIs while correcting only precomputed TQ views."""

    def __init__(
        self,
        torch_module: Any,
        plans: dict[tuple[int, tuple[int, ...]], _TQStridePlan],
    ) -> None:
        self._torch = torch_module
        self._plans = plans

    def __getattr__(self, name: str) -> Any:
        return getattr(self._torch, name)

    def as_strided(
        self,
        input: torch.Tensor,
        size: tuple[int, ...],
        stride: tuple[int, ...],
        storage_offset: int | None = None,
    ) -> torch.Tensor:
        plan = self._plans.get((input.data_ptr(), tuple(size)))
        if plan is None:
            if storage_offset is None:
                return self._torch.as_strided(input, size, stride)
            return self._torch.as_strided(
                input, size, stride, storage_offset=storage_offset
            )

        new_stride = list(stride)
        dim = plan.physical_block_dim
        if new_stride[dim] != plan.buggy_stride_elements:
            raise AssertionError(
                "TurboQuant padded-stride patch did not find the expected "
                f"legacy stride {plan.buggy_stride_elements}; got {new_stride[dim]}."
            )
        new_stride[dim] = plan.kernel_stride_elements

        offset = input.storage_offset() if storage_offset is None else storage_offset
        required_elements = offset + 1
        for dim_size, dim_stride in zip(size, new_stride):
            if dim_size > 0:
                required_elements += (dim_size - 1) * dim_stride
        available_elements = input.untyped_storage().nbytes() // input.element_size()
        if required_elements > available_elements:
            raise RuntimeError(
                "DFlash2 TurboQuant corrected page-strided view still exceeds "
                f"backing storage: need {required_elements * input.element_size()} "
                f"bytes, have {input.untyped_storage().nbytes()} bytes."
            )

        logger.info_once(
            "DFlash2 TurboQuant KV view splits each %d-byte logical padded page "
            "across %d virtual kernel blocks with %d-byte kernel stride.",
            plan.logical_page_bytes,
            plan.virtual_block_ratio,
            plan.kernel_stride_bytes,
        )
        if storage_offset is None:
            return self._torch.as_strided(input, size, tuple(new_stride))
        return self._torch.as_strided(
            input,
            size,
            tuple(new_stride),
            storage_offset=storage_offset,
        )


def _install_tq_padded_reshape() -> None:
    """Patch GPUModelRunner's padded TQ view without changing other models."""
    global _TQ_PADDED_RESHAPE_INSTALLED
    if _TQ_PADDED_RESHAPE_INSTALLED:
        return

    runner_mod = importlib.import_module("vllm.v1.worker.gpu_model_runner")
    runner_cls = runner_mod.GPUModelRunner
    original = runner_cls._reshape_kv_cache_tensors
    if getattr(original, "_dflash2_tq_padded_stride_compat", False):
        _TQ_PADDED_RESHAPE_INSTALLED = True
        return

    @wraps(original)
    def reshape_with_tq_kernel_stride(
        self,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
        kernel_block_sizes: list[int],
    ):
        plans = _build_tq_stride_plans(self, kv_cache_raw_tensors, kernel_block_sizes)
        if not plans:
            return original(self, kv_cache_raw_tensors, kernel_block_sizes)

        # GPUModelRunner imports torch as a module global. Swap only that module
        # reference while the single-threaded KV initialization reshape runs;
        # do not mutate torch.as_strided process-wide.
        with _RESHAPE_PATCH_LOCK:
            original_torch = runner_mod.torch
            runner_mod.torch = _TorchAsStridedProxy(original_torch, plans)
            try:
                return original(self, kv_cache_raw_tensors, kernel_block_sizes)
            finally:
                runner_mod.torch = original_torch

    reshape_with_tq_kernel_stride._dflash2_tq_padded_stride_compat = True  # type: ignore[attr-defined]
    runner_cls._reshape_kv_cache_tensors = reshape_with_tq_kernel_stride
    _TQ_PADDED_RESHAPE_INSTALLED = True


def install_dflash2_worker_kv_compat() -> None:
    """Install all Worker-side pieces needed by heterogeneous DFlash2 KV."""
    _install_tq_padded_reshape()

    # Store kernels execute in Workers as well. Import lazily to avoid a module
    # cycle at import time and keep EngineCore-only processes lightweight.
    from vllm.v1.core.dflash_kv_compat import _install_turboquant_strided_store

    _install_turboquant_strided_store()
