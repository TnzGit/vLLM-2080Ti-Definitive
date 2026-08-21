# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.core.dflash2_flex_attention_compat import (
    _select_legal_direct_kernel_options,
)


def test_kernel_tile_fallback_does_not_mutate_logical_metadata() -> None:
    metadata = SimpleNamespace(
        block_size=2160,
        kv_block_size=2160,
        direct_build=True,
    )

    def fake_original(query, block_m, block_n, use_direct_build):
        return {"BLOCK_M": 16, "BLOCK_N": 16}

    query = SimpleNamespace(dtype=torch.float16)
    options = _select_legal_direct_kernel_options(
        fake_original,
        query,
        block_m=16,
        block_n=metadata.kv_block_size,
        use_direct_build=metadata.direct_build,
    )

    assert metadata.block_size == 2160
    assert metadata.kv_block_size == 2160
    assert metadata.direct_build is True
    assert options["BLOCK_N"] == 16


def test_illegal_fallback_tile_is_rejected() -> None:
    def fake_original(query, block_m, block_n, use_direct_build):
        return {"BLOCK_M": 16, "BLOCK_N": 30}

    query = SimpleNamespace(dtype=torch.float16)
    with pytest.raises(RuntimeError, match="power-of-two Triton tiles"):
        _select_legal_direct_kernel_options(
            fake_original,
            query,
            block_m=16,
            block_n=2160,
            use_direct_build=True,
        )
