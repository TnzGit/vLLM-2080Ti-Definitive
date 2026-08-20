# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Install DFlash2 KV compatibility in the EngineCore process."""

from __future__ import annotations

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


def is_dflash2_config(vllm_config: VllmConfig) -> bool:
    """Return whether this engine is configured for the DFlash2 draft model."""
    speculative_config = vllm_config.speculative_config
    if speculative_config is None or speculative_config.method != "dflash":
        return False

    draft_model_config = speculative_config.draft_model_config
    hf_config = getattr(draft_model_config, "hf_config", None)
    if hf_config is None:
        return False

    architectures = getattr(hf_config, "architectures", ()) or ()
    if "DFlash2DraftModel" in architectures:
        return True

    dflash_config = getattr(hf_config, "dflash_config", {}) or {}
    return all(
        key in dflash_config
        for key in ("selector_rank", "selector_top_k", "conv_kernel_size")
    )


def maybe_install_dflash2_enginecore_kv_compat(vllm_config: VllmConfig) -> bool:
    """Install the heterogeneous-KV shim before EngineCore KV planning."""
    if not is_dflash2_config(vllm_config):
        return False

    from vllm.v1.core.dflash_kv_compat import (
        install_dflash2_heterogeneous_kv_compat,
    )

    install_dflash2_heterogeneous_kv_compat()
    logger.info_once(
        "Installed DFlash2 heterogeneous KV compatibility in the EngineCore "
        "process before KV-cache grouping."
    )
    return True
