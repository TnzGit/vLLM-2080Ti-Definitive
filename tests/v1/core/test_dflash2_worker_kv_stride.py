# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.core.dflash_kv_worker_compat import (
    _TorchAsStridedProxy,
    _tq_kernel_page_stride_bytes,
    _TQStridePlan,
)
from vllm.v1.kv_cache_interface import TQFullAttentionSpec


def test_reported_k8v4_virtual_block_stride_geometry() -> None:
    # Geometry reproduced from the real 2x RTX 2080 Ti failure:
    #   TQ real logical page:       1,676,160 B
    #   DFlash/common padded page:  4,423,680 B
    #   logical block_size:         2,160 tokens
    #   TQ kernel block_size:          16 tokens
    #   split ratio:                  135
    spec = TQFullAttentionSpec(
        block_size=2160,
        num_kv_heads=1,
        head_size=1,
        head_size_v=1,
        dtype=torch.uint8,
        tq_slot_size=776,
        page_size_padded=4_423_680,
    )
    assert spec.real_page_size_bytes == 1_676_160
    assert spec.page_size_bytes == 4_423_680

    kernel_block_size = 16
    ratio = spec.block_size // kernel_block_size
    assert ratio == 135
    kernel_stride = _tq_kernel_page_stride_bytes(spec, kernel_block_size)
    assert kernel_stride == 32_768

    num_logical_blocks = 136
    num_kernel_blocks = num_logical_blocks * ratio
    kernel_payload = spec.real_page_size_bytes // ratio
    assert num_kernel_blocks == 18_360
    assert kernel_payload == 12_416

    actual_storage = num_logical_blocks * spec.page_size_bytes
    buggy_required = (num_kernel_blocks - 1) * spec.page_size_bytes + kernel_payload
    corrected_required = (num_kernel_blocks - 1) * kernel_stride + kernel_payload

    assert actual_storage == 601_620_480
    assert buggy_required == 81_214_353_536
    assert corrected_required <= actual_storage


def test_as_strided_proxy_splits_padding_across_virtual_blocks() -> None:
    # Small CPU analogue of the same geometry: two 12-token logical pages,
    # split into three 4-token kernel blocks per logical page.
    shape = (6, 4, 1, 10)
    backing = torch.empty(2 * 240, dtype=torch.uint8)
    plan = _TQStridePlan(
        physical_shape=shape,
        physical_block_dim=0,
        buggy_stride_elements=240,
        kernel_stride_elements=80,
        logical_page_bytes=240,
        kernel_stride_bytes=80,
        virtual_block_ratio=3,
    )
    proxy = _TorchAsStridedProxy(torch, {(backing.data_ptr(), shape): plan})

    view = proxy.as_strided(
        backing,
        size=shape,
        stride=(240, 10, 10, 1),
    )

    assert view.shape == shape
    assert view.stride() == (80, 10, 10, 1)
    assert view[1].data_ptr() - view[0].data_ptr() == 80
    assert view[3].data_ptr() - view[0].data_ptr() == 240
    assert view.untyped_storage().nbytes() == backing.untyped_storage().nbytes()
