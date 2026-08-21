# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from functools import wraps

from vllm.v1.core.dflash_kv_bootstrap import (
    maybe_install_dflash2_enginecore_kv_compat,
)
from vllm.v1.core.dflash_private_kv import filter_dflash2_private_kv_specs

from .abstract import Executor
from .uniproc_executor import UniProcExecutor


# EngineCore calls Executor.get_kv_cache_specs() immediately before
# get_kv_cache_configs(). DFlash2's model loader runs only in worker processes,
# so install the planner shim here, in the EngineCore process, before the worker
# KV specs are handed to the central KV-cache planner. In the optional private
# DFlash KV mode, remove the five draft attention specs after the RPC so the
# target model is planned exactly as it is without DFlash.
if not getattr(Executor.get_kv_cache_specs, "_dflash2_enginecore_kv_compat", False):
    _original_get_kv_cache_specs = Executor.get_kv_cache_specs

    @wraps(_original_get_kv_cache_specs)
    def _get_kv_cache_specs_with_dflash2_compat(self):
        is_dflash2 = maybe_install_dflash2_enginecore_kv_compat(self.vllm_config)
        kv_cache_specs = _original_get_kv_cache_specs(self)
        if not is_dflash2:
            return kv_cache_specs
        return filter_dflash2_private_kv_specs(self.vllm_config, kv_cache_specs)

    _get_kv_cache_specs_with_dflash2_compat._dflash2_enginecore_kv_compat = True
    Executor.get_kv_cache_specs = _get_kv_cache_specs_with_dflash2_compat


__all__ = ["Executor", "UniProcExecutor"]
