# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fork-local DFlash2 loader with legacy heterogeneous-KV compatibility."""

from vllm.v1.core.dflash_kv_compat import install_dflash2_heterogeneous_kv_compat
from vllm.v1.core.dflash_kv_worker_compat import install_dflash2_worker_kv_compat

# The central KV planner runs in EngineCore, while the actual cache reshape and
# TurboQuant store kernels execute in worker processes. The Executor bootstrap
# installs the planner side before KV grouping; the DFlash2 model loader installs
# the worker-side virtual-block stride/store compatibility here.
install_dflash2_heterogeneous_kv_compat()
install_dflash2_worker_kv_compat()

from .qwen3_dflash2 import DFlash2Qwen3ForCausalLM  # noqa: E402

EntryClass = DFlash2Qwen3ForCausalLM

__all__ = ["DFlash2Qwen3ForCausalLM", "EntryClass"]
