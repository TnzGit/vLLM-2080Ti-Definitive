# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.v1.core.dflash2_flex_attention_compat import (
    _select_legal_direct_kernel_options,
)


def test_dflash2_direct_block_2160_uses_power_of_two_kernel_tile() -> None:
    calls: list[bool] = []

    def fake_original(query, block_m, block_n, use_direct_build):
        calls.append(use_direct_build)
        if use_direct_build:
            return {"BLOCK_M": block_m, "BLOCK_N": block_n}
        # Mirrors the existing general selector for the reported 2160 path.
        return {"BLOCK_M": 16, "BLOCK_N": 16}

    query = SimpleNamespace(dtype=torch.float16)
    options = _select_legal_direct_kernel_options(
        fake_original,
        query,
        block_m=16,
        block_n=2160,
        use_direct_build=True,
    )

    assert calls == [False]
    assert options["BLOCK_M"] == 16
    assert options["BLOCK_N"] == 16
    assert 2160 % int(options["BLOCK_N"]) == 0


def test_power_of_two_direct_geometry_keeps_native_fast_path() -> None:
    calls: list[bool] = []

    def fake_original(query, block_m, block_n, use_direct_build):
        calls.append(use_direct_build)
        return {"BLOCK_M": block_m, "BLOCK_N": block_n}

    query = SimpleNamespace(dtype=torch.float16)
    options = _select_legal_direct_kernel_options(
        fake_original,
        query,
        block_m=16,
        block_n=2048,
        use_direct_build=True,
    )

    assert calls == [True]
    assert options == {"BLOCK_M": 16, "BLOCK_N": 2048}
