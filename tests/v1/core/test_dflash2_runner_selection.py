# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.config import VllmConfig


def _config(*, method="dflash", architectures=(), dflash_config=None):
    hf_config = SimpleNamespace(
        architectures=list(architectures),
        dflash_config={} if dflash_config is None else dflash_config,
    )
    draft_model_config = SimpleNamespace(hf_config=hf_config)
    speculative_config = SimpleNamespace(
        method=method,
        draft_model_config=draft_model_config,
    )
    return SimpleNamespace(speculative_config=speculative_config)


def test_dflash2_draft_forces_v1_runner_by_architecture() -> None:
    config = _config(architectures=("DFlash2DraftModel",))
    assert VllmConfig._dflash2_needs_v1_model_runner(config)


def test_dflash2_draft_forces_v1_runner_by_selector_config() -> None:
    config = _config(
        dflash_config={
            "selector_rank": 64,
            "selector_top_k": 8,
            "conv_kernel_size": 3,
        }
    )
    assert VllmConfig._dflash2_needs_v1_model_runner(config)


def test_dflash1_draft_does_not_force_v1_runner() -> None:
    assert not VllmConfig._dflash2_needs_v1_model_runner(
        _config(architectures=("DFlashDraftModel",))
    )
    assert not VllmConfig._dflash2_needs_v1_model_runner(
        _config(
            architectures=("DFlashDraftModel",),
            dflash_config={"conv_kernel_size": 3},
        )
    )


def test_non_dflash_methods_do_not_force_v1_runner() -> None:
    assert not VllmConfig._dflash2_needs_v1_model_runner(
        _config(method="mtp", architectures=("DFlash2DraftModel",))
    )
    assert not VllmConfig._dflash2_needs_v1_model_runner(
        SimpleNamespace(speculative_config=None)
    )
