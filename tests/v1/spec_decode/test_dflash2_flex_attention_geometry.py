# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from vllm.v1.core.dflash2_flex_attention_compat import _fix_flex_block_geometry


def test_dflash2_flex_block_geometry_rounds_to_power_of_two():
    metadata = SimpleNamespace(kv_block_size=2160, block_mask="old")

    _fix_flex_block_geometry(metadata)

    assert metadata.kv_block_size == 4096
    assert metadata.block_mask is None


def test_dflash2_flex_block_geometry_keeps_power_of_two():
    metadata = SimpleNamespace(kv_block_size=2048, block_mask="old")

    _fix_flex_block_geometry(metadata)

    assert metadata.kv_block_size == 2048
    assert metadata.block_mask == "old"
