# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.utils import WeightsMapper
from vllm.plugins.gguf_sm75 import (
    GGUFHeadTilingLayout,
    SM75GGUFConfig,
    _gdn_value_head_layout,
    _require_first_stage_scope,
    _transform_qwen35_weight,
    build_qwen35_name_map,
)


def test_head_tiling_layout_round_trip() -> None:
    layout = GGUFHeadTilingLayout(heads_per_group=2, head_dim=2)
    original = torch.arange(16).reshape(2, 8)

    stored = layout.input_to_gguf(original)
    restored = layout.weight_to_vllm(stored, dim=1)

    torch.testing.assert_close(restored, original)
    assert not torch.equal(stored, original)


def test_head_tiling_layout_shards_packed_dimension_by_group() -> None:
    layout = GGUFHeadTilingLayout(heads_per_group=2, head_dim=4)
    weight = torch.arange(48).reshape(3, 16)

    rank0 = layout.shard_weight(
        weight,
        dim=1,
        logical_size=32,
        block_size=4,
        tp_rank=0,
        tp_size=2,
    )
    rank1 = layout.shard_weight(
        weight,
        dim=1,
        logical_size=32,
        block_size=4,
        tp_rank=1,
        tp_size=2,
    )

    moved = weight.movedim(1, 0).reshape(2, 4, 2, 3)
    expected0 = moved[:, :2].reshape(-1, 3).movedim(0, 1)
    expected1 = moved[:, 2:].reshape(-1, 3).movedim(0, 1)
    torch.testing.assert_close(rank0, expected0)
    torch.testing.assert_close(rank1, expected1)


def test_head_tiling_layout_rejects_tp_split_inside_ggml_block() -> None:
    layout = GGUFHeadTilingLayout(heads_per_group=2, head_dim=4)
    weight = torch.arange(48).reshape(3, 16)

    with pytest.raises(ValueError, match="not aligned to GGML block size"):
        layout.shard_weight(
            weight,
            dim=1,
            logical_size=32,
            block_size=16,
            tp_rank=0,
            tp_size=2,
        )


def test_qwen35_text_name_map_for_nested_qwen36_config() -> None:
    mapping = build_qwen35_name_map(
        [
            "token_embd.weight",
            "blk.0.attn_norm.weight",
            "blk.0.attn_qkv.weight",
            "blk.0.attn_gate.weight",
            "blk.0.ssm_alpha.weight",
            "blk.0.ssm_beta.weight",
            "blk.0.ssm_conv1d.weight",
            "blk.0.ssm_norm.weight",
            "blk.0.ssm_out.weight",
            "blk.0.ssm_dt.bias",
            "blk.0.ssm_a",
            "blk.0.ffn_gate.weight",
            "blk.0.ffn_up.weight",
            "blk.0.ffn_down.weight",
            "output_norm.weight",
            "output.weight",
        ],
        nested_multimodal_config=True,
    )

    assert mapping["token_embd.weight"] == "model.language_model.embed_tokens.weight"
    assert (
        mapping["blk.0.attn_qkv.weight"]
        == "model.language_model.layers.0.linear_attn.in_proj_qkv.weight"
    )
    assert (
        mapping["blk.0.ssm_out.weight"]
        == "model.language_model.layers.0.linear_attn.out_proj.weight"
    )
    assert (
        mapping["blk.0.ffn_gate.weight"]
        == "model.language_model.layers.0.mlp.gate_proj.weight"
    )
    assert mapping["output.weight"] == "lm_head.weight"


def test_qwen35_mapper_skips_mtp_and_multimodal_sidecar_tensors() -> None:
    mapping = build_qwen35_name_map(
        [
            "token_embd.weight",
            "blk.64.nextn.eh_proj.weight",
            "v.blk.0.attn_qkv.weight",
            "mm.0.weight",
        ],
        nested_multimodal_config=True,
    )

    assert mapping == {"token_embd.weight": "model.language_model.embed_tokens.weight"}


def test_qwen35_gdn_layout_detects_grouped_value_heads() -> None:
    config = SimpleNamespace(
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_value_head_dim=128,
    )
    layout = _gdn_value_head_layout(config)

    assert layout == GGUFHeadTilingLayout(heads_per_group=2, head_dim=128)


def test_qwen35_gdn_transforms_norm_and_state_weights() -> None:
    config = SimpleNamespace(
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
    )
    layout = _gdn_value_head_layout(config)
    assert layout is not None

    norm = torch.tensor([1.5, 2.5])
    transformed_norm = _transform_qwen35_weight(
        "model.layers.0.input_layernorm.weight", norm, config, layout
    )
    torch.testing.assert_close(transformed_norm, torch.tensor([0.5, 1.5]))

    stored_a = -torch.tensor([1.0, 3.0, 2.0, 4.0])
    transformed_a = _transform_qwen35_weight(
        "model.layers.0.linear_attn.A_log", stored_a, config, layout
    )
    expected = layout.weight_to_vllm(torch.log(-stored_a), dim=0, head_dim=1)
    torch.testing.assert_close(transformed_a, expected)


def test_qwen35_qkv_transform_only_reorders_value_rows() -> None:
    config = SimpleNamespace(
        linear_num_key_heads=2,
        linear_num_value_heads=4,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
    )
    layout = _gdn_value_head_layout(config)
    assert layout is not None

    # q + k = 8 rows. The final 8 value rows are in GGML head-tile order.
    weight = torch.arange(16 * 3).reshape(16, 3)
    transformed = _transform_qwen35_weight(
        "model.layers.0.linear_attn.in_proj_qkv.qweight",
        weight,
        config,
        layout,
    )
    torch.testing.assert_close(transformed[:8], weight[:8])
    torch.testing.assert_close(
        transformed[8:], layout.weight_to_vllm(weight[8:], dim=0)
    )


def test_quant_config_maps_registered_layout_through_model_mapper() -> None:
    config = SM75GGUFConfig()
    layout = GGUFHeadTilingLayout(heads_per_group=2, head_dim=128)
    config.register_linear_layouts(
        {"model.language_model.layers.0.linear_attn.out_proj": layout}
    )

    mapper = WeightsMapper(
        orig_to_new_prefix={"model.language_model.": "language_model.model."}
    )
    config.apply_vllm_mapper(mapper)

    assert config.linear_layouts == {
        "language_model.model.layers.0.linear_attn.out_proj": layout
    }


def test_first_stage_requires_language_model_only_for_nested_qwen35() -> None:
    text_config = SimpleNamespace(model_type="qwen3_5_text")
    hf_config = SimpleNamespace(model_type="qwen3_5", vision_config=object())
    model_config = SimpleNamespace(
        hf_config=hf_config,
        hf_text_config=text_config,
        multimodal_config=SimpleNamespace(language_model_only=False),
    )

    with pytest.raises(NotImplementedError, match="--language-model-only"):
        _require_first_stage_scope(model_config)

    model_config.multimodal_config.language_model_only = True
    _require_first_stage_scope(model_config)
