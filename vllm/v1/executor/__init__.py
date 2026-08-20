# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import wraps

from vllm.v1.core.dflash_kv_bootstrap import (
    maybe_install_dflash2_enginecore_kv_compat,
)

from .abstract import Executor
from .uniproc_executor import UniProcExecutor


# EngineCore calls Executor.get_kv_cache_specs() immediately before
# get_kv_cache_configs(). DFlash2's model loader runs only in worker processes,
# so install the planner shim here, in the EngineCore process, before the worker
# KV specs are handed to the central KV-cache planner.
if not getattr(Executor.get_kv_cache_specs, "_dflash2_enginecore_kv_compat", False):
    _original_get_kv_cache_specs = Executor.get_kv_cache_specs

    @wraps(_original_get_kv_cache_specs)
    def _get_kv_cache_specs_with_dflash2_compat(self):
        maybe_install_dflash2_enginecore_kv_compat(self.vllm_config)
        return _original_get_kv_cache_specs(self)

    _get_kv_cache_specs_with_dflash2_compat._dflash2_enginecore_kv_compat = True
    Executor.get_kv_cache_specs = _get_kv_cache_specs_with_dflash2_compat


__all__ = ["Executor", "UniProcExecutor"]
