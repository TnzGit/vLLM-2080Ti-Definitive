# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fork-local DFlash2 loader with legacy heterogeneous-KV compatibility."""

from vllm.v1.core.dflash_kv_compat import install_dflash2_heterogeneous_kv_compat

install_dflash2_heterogeneous_kv_compat()

from .qwen3_dflash2 import DFlash2Qwen3ForCausalLM  # noqa: E402

EntryClass = DFlash2Qwen3ForCausalLM

__all__ = ["DFlash2Qwen3ForCausalLM", "EntryClass"]
