# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.core.dflash_kv_compat import _pad_nondivisible_page_sizes
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    TQFullAttentionSpec,
)


def test_dflash2_nondivisible_pages_use_physical_padding() -> None:
    block_size = 16
    tq = TQFullAttentionSpec(
        block_size=block_size,
        num_kv_heads=2,
        head_size=7,
        head_size_v=7,
        dtype=torch.float16,
        tq_slot_size=13,
    )
    dflash = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=2,
        head_size=7,
        head_size_v=7,
        dtype=torch.float16,
    )
    mamba = MambaSpec(
        block_size=block_size,
        shapes=((3,),),
        dtypes=(torch.float16,),
    )

    assert tq.real_page_size_bytes == 416
    assert dflash.real_page_size_bytes == 896
    assert dflash.page_size_bytes % tq.page_size_bytes != 0

    padded = _pad_nondivisible_page_sizes(
        {"target.tq": tq, "draft.dflash": dflash, "target.mamba": mamba}
    )

    assert {spec.page_size_bytes for spec in padded.values()} == {896}
    assert all(spec.block_size == block_size for spec in padded.values())
    assert padded["target.tq"].real_page_size_bytes == 416
    assert padded["draft.dflash"].real_page_size_bytes == 896

    # The helper is non-destructive: the target's logical/real cache geometry
    # remains unchanged outside the returned compatibility specs.
    assert tq.page_size_bytes == 416
    assert mamba.page_size_bytes == 6


def test_page_padding_produces_noncontiguous_block_stride() -> None:
    num_blocks = 2
    block_size = 16
    num_heads = 2
    slot_size = 13
    real_page = block_size * num_heads * slot_size
    padded_page = 896
    assert real_page == 416

    backing = torch.empty(num_blocks * padded_page, dtype=torch.uint8)
    kv_cache = torch.as_strided(
        backing,
        size=(num_blocks, block_size, num_heads, slot_size),
        stride=(padded_page, num_heads * slot_size, slot_size, 1),
    )

    assert not kv_cache.is_contiguous()
    assert kv_cache.stride(0) == padded_page
    assert kv_cache[1].data_ptr() - kv_cache[0].data_ptr() == padded_page
