# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.v1.core.dflash2_flex_attention_compat import _fix_flex_block_geometry


def test_dflash2_flex_geometry_does_not_expand_to_4096():
    metadata = SimpleNamespace(
        kv_block_size=2160,
        block_mask=object(),
    )

    _fix_flex_block_geometry(metadata)

    assert metadata.kv_block_size == 2048
    assert metadata.block_mask is None


def test_power_of_two_geometry_is_unchanged():
    metadata = SimpleNamespace(
        kv_block_size=1024,
        block_mask=object(),
    )

    _fix_flex_block_geometry(metadata)

    assert metadata.kv_block_size == 1024
