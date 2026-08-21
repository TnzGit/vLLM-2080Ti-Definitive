# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.core.dflash_private_kv import (
    _advance_window_state,
    filter_dflash2_private_kv_specs,
    get_dflash_private_kv_window,
)
from vllm.v1.core.dflash_private_kv_anchor import (
    select_private_metadata_anchor_gid,
)


class _Spec:
    def __init__(self, page_size: int):
        self.page_size_bytes = page_size
        self.real_page_size_bytes = page_size


def _config(target_layers: int = 64, draft_layers: int = 5):
    model_config = SimpleNamespace(
        get_num_layers=lambda parallel_config: target_layers,
    )
    draft_hf = SimpleNamespace(num_hidden_layers=draft_layers)
    speculative_config = SimpleNamespace(
        draft_model_config=SimpleNamespace(hf_config=draft_hf)
    )
    return SimpleNamespace(
        model_config=model_config,
        parallel_config=SimpleNamespace(),
        speculative_config=speculative_config,
    )


@pytest.fixture(autouse=True)
def _clear_private_window_cache(monkeypatch):
    get_dflash_private_kv_window.cache_clear()
    monkeypatch.delenv("VLLM_DFLASH_PRIVATE_KV_WINDOW", raising=False)
    yield
    get_dflash_private_kv_window.cache_clear()


def test_private_window_disabled_by_default():
    assert get_dflash_private_kv_window() == 0


def test_private_window_parser(monkeypatch):
    monkeypatch.setenv("VLLM_DFLASH_PRIVATE_KV_WINDOW", "16384")
    get_dflash_private_kv_window.cache_clear()
    assert get_dflash_private_kv_window() == 16384


def test_private_filter_removes_only_dflash_layers(monkeypatch):
    monkeypatch.setenv("VLLM_DFLASH_PRIVATE_KV_WINDOW", "16384")
    get_dflash_private_kv_window.cache_clear()

    specs = {
        "model.layers.5.self_attn.attn": _Spec(111),
        "model.layers.63.self_attn.attn": _Spec(222),
        **{f"model.layers.{i}.self_attn.attn": _Spec(4_423_680) for i in range(64, 69)},
    }
    result = filter_dflash2_private_kv_specs(_config(), [specs])

    assert len(result) == 1
    assert set(result[0]) == {
        "model.layers.5.self_attn.attn",
        "model.layers.63.self_attn.attn",
    }
    assert set(specs) != set(result[0])  # input dict is not modified


def test_private_filter_fails_closed_when_draft_specs_are_missing(monkeypatch):
    monkeypatch.setenv("VLLM_DFLASH_PRIVATE_KV_WINDOW", "16384")
    get_dflash_private_kv_window.cache_clear()

    specs = {
        f"model.layers.{i}.self_attn.attn": _Spec(4_423_680) for i in range(64, 68)
    }
    with pytest.raises(
        RuntimeError, match="could not identify all draft attention specs"
    ):
        filter_dflash2_private_kv_specs(_config(), [specs])


def test_private_metadata_anchor_uses_first_nonempty_target_group():
    config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=[]),
            SimpleNamespace(layer_names=["model.layers.0.linear_attn"]),
            SimpleNamespace(layer_names=["model.layers.5.self_attn.attn"]),
        ]
    )
    assert select_private_metadata_anchor_gid(config) == 1


def test_private_metadata_anchor_fails_without_target_group():
    config = SimpleNamespace(
        kv_cache_groups=[
            SimpleNamespace(layer_names=[]),
            SimpleNamespace(layer_names=[]),
        ]
    )
    with pytest.raises(RuntimeError, match="at least one managed target KV group"):
        select_private_metadata_anchor_gid(config)


def test_window_state_extends_and_rolls():
    start, end = _advance_window_state(0, -1, 0, 99, 128)
    assert (start, end) == (0, 99)

    start, end = _advance_window_state(start, end, 100, 199, 128)
    assert (start, end) == (72, 199)

    start, end = _advance_window_state(start, end, 196, 205, 128)
    assert (start, end) == (78, 205)


def test_window_state_resets_on_new_request_or_gap():
    assert _advance_window_state(72, 199, 0, 9, 128) == (0, 9)
    assert _advance_window_state(72, 199, 1000, 1010, 128) == (1000, 1010)
