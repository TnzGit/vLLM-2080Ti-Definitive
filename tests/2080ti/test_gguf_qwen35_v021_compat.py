# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
from torch import nn

from vllm.plugins.gguf_sm75_stage1 import (
    _detach_staged_gguf_views,
    _gguf_config_source,
    _is_gguf_reference,
    _storage_nbytes,
    _tuple_and_layout_aware_weight_loader,
    _vocab_params_dtype,
)


def test_detach_staged_gguf_view_releases_full_backing_storage() -> None:
    full = torch.arange(16 * 4, dtype=torch.uint8).reshape(16, 4)
    staged = full.narrow(0, 0, 8)
    param = SimpleNamespace(is_gguf_weight=True, data_container=[staged])

    full_bytes = full.numel() * full.element_size()
    local_bytes = staged.numel() * staged.element_size()
    assert _storage_nbytes(staged) == full_bytes

    avoided, retained = _detach_staged_gguf_views(param)
    detached = param.data_container[0]

    assert avoided == full_bytes - local_bytes
    assert retained == local_bytes
    assert _storage_nbytes(detached) == local_bytes
    assert detached.untyped_storage().data_ptr() != full.untyped_storage().data_ptr()
    torch.testing.assert_close(detached, full[:8])


def test_detach_staged_gguf_views_counts_shared_backing_once() -> None:
    full = torch.arange(16 * 4, dtype=torch.uint8).reshape(16, 4)
    staged = [
        full.narrow(0, 0, 2),
        full.narrow(0, 4, 2),
        full.narrow(0, 8, 4),
    ]
    param = SimpleNamespace(is_gguf_weight=True, data_container=staged)

    full_bytes = full.numel() * full.element_size()
    local_bytes = sum(t.numel() * t.element_size() for t in staged)

    avoided, retained = _detach_staged_gguf_views(param)

    assert avoided == full_bytes - local_bytes
    assert retained == local_bytes
    for detached in param.data_container:
        assert _storage_nbytes(detached) == detached.numel() * detached.element_size()
        assert detached.untyped_storage().data_ptr() != full.untyped_storage().data_ptr()


def test_detach_staged_gguf_views_leaves_compact_owner_untouched() -> None:
    compact = torch.arange(8, dtype=torch.uint8)
    param = SimpleNamespace(is_gguf_weight=True, data_container=[compact])
    storage_ptr = compact.untyped_storage().data_ptr()

    avoided, retained = _detach_staged_gguf_views(param)

    assert avoided == 0
    assert retained == 0
    assert param.data_container[0].untyped_storage().data_ptr() == storage_ptr


def test_tuple_gguf_weight_splits_fused_qkv_into_logical_shards() -> None:
    layer = SimpleNamespace(output_sizes=[4, 4, 8, 8])
    param = SimpleNamespace(
        is_gguf_weight=True,
        is_gguf_weight_type=False,
        output_dim=0,
        data_container=[],
        shard_id=[],
        shard_id_map={},
    )
    calls: list[tuple[int | None, torch.Tensor]] = []

    def fallback(_param, loaded_weight, shard_id=None):
        calls.append((shard_id, loaded_weight.clone()))

    loaded = torch.arange(16 * 3).reshape(16, 3)
    _tuple_and_layout_aware_weight_loader(
        layer,
        fallback,
        None,
        0,
        param,
        loaded,
        (0, 1, 2),
    )

    assert [shard_id for shard_id, _ in calls] == [0, 1, 2]
    torch.testing.assert_close(calls[0][1], loaded[0:4])
    torch.testing.assert_close(calls[1][1], loaded[4:8])
    torch.testing.assert_close(calls[2][1], loaded[8:16])


def test_tuple_gguf_weight_detaches_tp_local_views_after_group_load() -> None:
    layer = SimpleNamespace(output_sizes=[4, 4, 8, 8])
    param = SimpleNamespace(
        is_gguf_weight=True,
        is_gguf_weight_type=False,
        output_dim=0,
        data_container=[],
        shard_id=[],
        shard_id_map={},
    )

    def fallback(target, loaded_weight, shard_id=None):
        # Model the vLLM 0.21 TP=2 path: the fallback loader narrows the
        # logical shard to this rank and stores a view in data_container.
        local = loaded_weight.narrow(0, 0, loaded_weight.shape[0] // 2)
        target.shard_id_map[shard_id] = len(target.data_container)
        target.shard_id.append(shard_id)
        target.data_container.append(local)

    loaded = torch.arange(16 * 3, dtype=torch.int64).reshape(16, 3)
    original_storage_ptr = loaded.untyped_storage().data_ptr()
    _tuple_and_layout_aware_weight_loader(
        layer,
        fallback,
        None,
        0,
        param,
        loaded,
        (0, 1, 2),
    )

    assert param.shard_id == [0, 1, 2]
    assert [tuple(t.shape) for t in param.data_container] == [(2, 3), (2, 3), (4, 3)]
    for staged in param.data_container:
        assert staged.untyped_storage().data_ptr() != original_storage_ptr
        assert _storage_nbytes(staged) == staged.numel() * staged.element_size()


def test_tuple_gguf_weight_type_is_repeated_for_each_logical_shard() -> None:
    layer = SimpleNamespace(output_sizes=[4, 4, 8, 8])
    param = SimpleNamespace(
        is_gguf_weight=False,
        is_gguf_weight_type=True,
    )
    calls: list[tuple[int | None, int]] = []

    def fallback(_param, loaded_weight, shard_id=None):
        calls.append((shard_id, int(loaded_weight.item())))

    loaded_type = torch.tensor(12, dtype=torch.uint8)
    _tuple_and_layout_aware_weight_loader(
        layer,
        fallback,
        None,
        0,
        param,
        loaded_type,
        (0, 1, 2),
    )

    assert calls == [(0, 12), (1, 12), (2, 12)]


def test_standard_qkv_string_shard_id_passes_through_unchanged() -> None:
    layer = SimpleNamespace(output_sizes=[4, 4, 8])
    param = SimpleNamespace(
        is_gguf_weight=True,
        is_gguf_weight_type=False,
        output_dim=0,
        data_container=[],
    )
    calls: list[tuple[str | None, torch.Tensor]] = []

    def fallback(_param, loaded_weight, shard_id=None):
        calls.append((shard_id, loaded_weight.clone()))

    loaded = torch.arange(12).reshape(4, 3)
    _tuple_and_layout_aware_weight_loader(
        layer,
        fallback,
        None,
        0,
        param,
        loaded,
        "q",
    )

    assert len(calls) == 1
    assert calls[0][0] == "q"
    torch.testing.assert_close(calls[0][1], loaded)


def test_tuple_gguf_weight_requires_exact_fused_output_size() -> None:
    layer = SimpleNamespace(output_sizes=[4, 4, 8, 8])
    param = SimpleNamespace(
        is_gguf_weight=True,
        is_gguf_weight_type=False,
        output_dim=0,
        data_container=[],
    )

    def fallback(_param, _loaded_weight, shard_id=None):
        del shard_id

    loaded = torch.arange(15 * 3).reshape(15, 3)
    try:
        _tuple_and_layout_aware_weight_loader(
            layer,
            fallback,
            None,
            0,
            param,
            loaded,
            (0, 1, 2),
        )
    except RuntimeError:
        # torch.narrow rejects the final 8-row slice before the explicit size
        # check; either failure mode proves malformed fused tensors fail closed.
        pass
    else:
        raise AssertionError("Malformed fused GGUF tensor should be rejected")


def test_vocab_params_dtype_falls_back_to_existing_parameter() -> None:
    module = nn.Module()
    module.register_parameter(
        "weight",
        nn.Parameter(torch.empty(2, 2, dtype=torch.float16), requires_grad=False),
    )

    assert _vocab_params_dtype(module) is torch.float16


def test_vocab_params_dtype_prefers_explicit_attribute() -> None:
    module = nn.Module()
    module.params_dtype = torch.bfloat16
    module.register_parameter(
        "weight",
        nn.Parameter(torch.empty(2, 2, dtype=torch.float16), requires_grad=False),
    )

    assert _vocab_params_dtype(module) is torch.bfloat16


def test_gguf_config_source_prefers_explicit_hf_config_path() -> None:
    assert (
        _gguf_config_source(
            "unsloth/Qwen3.6-27B-GGUF:Q4_K_M",
            "Qwen/Qwen3.6-27B",
            "Qwen/Qwen3.6-27B-custom-config",
        )
        == "Qwen/Qwen3.6-27B-custom-config"
    )


def test_gguf_config_source_falls_back_to_non_gguf_tokenizer() -> None:
    assert (
        _gguf_config_source(
            "unsloth/Qwen3.6-27B-GGUF:Q4_K_M",
            "Qwen/Qwen3.6-27B",
            None,
        )
        == "Qwen/Qwen3.6-27B"
    )


def test_gguf_config_source_does_not_reuse_gguf_tokenizer_ref() -> None:
    gguf_ref = "unsloth/Qwen3.6-27B-GGUF:Q4_K_M"
    assert _gguf_config_source(gguf_ref, gguf_ref, None) is None


def test_gguf_reference_recognizes_repo_quant_and_exact_remote_file() -> None:
    assert _is_gguf_reference("unsloth/Qwen3.6-27B-GGUF:Q4_K_M")
    assert _is_gguf_reference(
        "unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf"
    )


def test_gguf_reference_rejects_plain_hf_model() -> None:
    assert not _is_gguf_reference("Qwen/Qwen3.6-27B")
