# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fork-local DFlash2 loader with legacy runtime compatibility."""

import torch

from vllm.v1.core.dflash_kv_compat import install_dflash2_heterogeneous_kv_compat
from vllm.v1.core.dflash_kv_worker_compat import install_dflash2_worker_kv_compat
from vllm.v1.core.dflash2_flex_attention_compat import (
    install_dflash2_flex_attention_compat,
)

# The central KV planner runs in EngineCore, while the actual cache reshape and
# TurboQuant store kernels execute in worker processes. The DFlash2 model loader
# installs worker-side compatibility hooks for the remaining execution paths.
install_dflash2_heterogeneous_kv_compat()
install_dflash2_worker_kv_compat()
install_dflash2_flex_attention_compat()

from .dflash2_dtype_compat import combine_dflash2_hidden_states  # noqa: E402
from .qwen3_dflash2 import (  # noqa: E402
    DFlash2Qwen3ForCausalLM as _DFlash2Qwen3ForCausalLM,
)


class DFlash2Qwen3ForCausalLM(_DFlash2Qwen3ForCausalLM):
    """Fork-local DFlash2 entry class with target/draft dtype normalization."""

    def combine_hidden_states(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return combine_dflash2_hidden_states(self, hidden_states)


EntryClass = DFlash2Qwen3ForCausalLM

__all__ = ["DFlash2Qwen3ForCausalLM", "EntryClass"]
