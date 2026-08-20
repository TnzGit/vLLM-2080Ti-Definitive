# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import vllm.v1.executor as executor_module
from vllm.v1.core.dflash_kv_bootstrap import is_dflash2_config
from vllm.v1.executor import Executor


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


def test_dflash2_enginecore_detection_by_architecture() -> None:
    assert is_dflash2_config(_config(architectures=("DFlash2DraftModel",)))
    assert not is_dflash2_config(_config(architectures=("DFlashDraftModel",)))
    assert not is_dflash2_config(
        _config(method="eagle", architectures=("DFlash2DraftModel",))
    )


def test_dflash2_enginecore_detection_by_selector_config() -> None:
    assert is_dflash2_config(
        _config(
            dflash_config={
                "selector_rank": 64,
                "selector_top_k": 8,
                "conv_kernel_size": 3,
            }
        )
    )


def test_executor_installs_compat_before_kv_spec_rpc(monkeypatch) -> None:
    events: list[str] = []

    def fake_install(vllm_config):
        assert is_dflash2_config(vllm_config)
        events.append("install")
        return True

    monkeypatch.setattr(
        executor_module,
        "maybe_install_dflash2_enginecore_kv_compat",
        fake_install,
    )

    sentinel = object()

    class FakeExecutor:
        vllm_config = _config(architectures=("DFlash2DraftModel",))

        def collective_rpc(self, method):
            assert method == "get_kv_cache_spec"
            events.append("rpc")
            return [{"layer": sentinel}]

    result = Executor.get_kv_cache_specs(FakeExecutor())

    assert result == [{"layer": sentinel}]
    assert events == ["install", "rpc"]
