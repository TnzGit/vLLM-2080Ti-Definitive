# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fork-local DFlash2 loader with legacy heterogeneous-KV compatibility."""

from vllm.v1.core.dflash_kv_compat import (
    _install_turboquant_strided_store,
    install_dflash2_heterogeneous_kv_compat,
)

# The central KV planner runs in EngineCore, but the actual TurboQuant store
# kernels execute in worker processes. Install both sides here when the DFlash2
# model class is loaded in a worker. The EngineCore-side planner installation is
# performed separately by vllm.v1.executor's pre-planning hook.
install_dflash2_heterogeneous_kv_compat()
_install_turboquant_strided_store()

from .qwen3_dflash2 import DFlash2Qwen3ForCausalLM  # noqa: E402

EntryClass = DFlash2Qwen3ForCausalLM

__all__ = ["DFlash2Qwen3ForCausalLM", "EntryClass"]
