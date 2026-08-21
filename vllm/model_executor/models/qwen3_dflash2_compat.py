# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fork-local DFlash2 loader with legacy runtime compatibility."""

import torch

from vllm.v1.core.dflash_private_kv import (
    dflash_private_kv_enabled,
    install_dflash2_private_kv_worker_compat,
)

# The central KV planner runs in EngineCore, while model/cache execution lives
# in worker processes. Keep the proven managed/padded path as the default.
# The opt-in private/windowed path deliberately bypasses those hooks so the
# target's native KV layout is not perturbed by DFlash at all.
if dflash_private_kv_enabled():
    install_dflash2_private_kv_worker_compat()
    # The legacy GPUModelRunner still chooses speculative CommonAttentionMetadata
    # by matching drafter.kv_cache_gid. Private KV has no managed draft group, so
    # retain one target group purely as a metadata anchor after detaching draft KV.
    from vllm.v1.core.dflash_private_kv_anchor import (
        install_dflash2_private_metadata_anchor,
    )

    install_dflash2_private_metadata_anchor()
else:
    from vllm.v1.core.dflash2_flex_attention_compat import (
        install_dflash2_flex_attention_compat,
    )
    from vllm.v1.core.dflash_kv_compat import install_dflash2_heterogeneous_kv_compat
    from vllm.v1.core.dflash_kv_worker_compat import install_dflash2_worker_kv_compat

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
