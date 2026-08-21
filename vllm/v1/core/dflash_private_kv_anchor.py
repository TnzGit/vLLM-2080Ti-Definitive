# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata-anchor compatibility for the experimental DFlash2 private KV path.

The legacy GPUModelRunner chooses the common speculative-decoding attention
metadata by matching ``drafter.kv_cache_gid`` against one of the centrally
managed KV groups.  Private DFlash KV deliberately removes the draft attention
layers from that manager, but the proposer still needs a CommonAttentionMetadata
instance for request/query lengths and positions.

This shim leaves draft KV fully private while pointing ``kv_cache_gid`` at the
first non-empty *target* group purely as a metadata anchor.  The private DFlash
attention path never reads that group's KV cache or block table.
"""

from __future__ import annotations

from functools import wraps
from typing import Any

from vllm.logger import init_logger

from .dflash_private_kv import dflash_private_kv_enabled

logger = init_logger(__name__)

_INSTALLED = False


def select_private_metadata_anchor_gid(kv_cache_config: Any) -> int:
    """Return the first target KV group that exists on this worker.

    Projected PP configs may contain empty groups, so do not blindly choose 0.
    The selected group is only used by GPUModelRunner to provide the drafter a
    CommonAttentionMetadata object; private DFlash never consumes its KV data.
    """
    groups = getattr(kv_cache_config, "kv_cache_groups", ()) or ()
    for gid, group in enumerate(groups):
        if getattr(group, "layer_names", None):
            return gid
    raise RuntimeError(
        "DFlash2 private KV requires at least one managed target KV group to "
        "anchor speculative-decoding CommonAttentionMetadata."
    )


def install_dflash2_private_metadata_anchor() -> bool:
    """Keep a target metadata anchor after private KV detaches draft groups."""
    global _INSTALLED
    if not dflash_private_kv_enabled():
        return False
    if _INSTALLED:
        return True

    from vllm.v1.spec_decode.dflash import DFlashProposer

    original = DFlashProposer.initialize_attn_backend
    if getattr(original, "_dflash_private_metadata_anchor", False):
        _INSTALLED = True
        return True

    @wraps(original)
    def initialize_with_metadata_anchor(
        self,
        kv_cache_config,
        kernel_block_sizes=None,
    ):
        result = original(self, kv_cache_config, kernel_block_sizes)
        if not getattr(self, "_is_dflash2", False):
            return result

        # The private worker shim must already have detached all draft groups.
        # If it did not, fail closed instead of silently mixing managed and
        # private DFlash KV ownership.
        if getattr(self, "draft_attn_groups", None):
            raise RuntimeError(
                "DFlash2 private KV metadata anchor found managed draft "
                "attention groups; refusing mixed KV ownership."
            )

        anchor_gid = select_private_metadata_anchor_gid(kv_cache_config)
        self.kv_cache_gid = anchor_gid
        logger.info_once(
            "DFlash2 private KV uses target KV group %d as the speculative "
            "CommonAttentionMetadata anchor; draft KV remains fully private.",
            anchor_gid,
        )
        return result

    initialize_with_metadata_anchor._dflash_private_metadata_anchor = True  # type: ignore[attr-defined]
    DFlashProposer.initialize_attn_backend = initialize_with_metadata_anchor
    _INSTALLED = True
    return True


__all__ = [
    "install_dflash2_private_metadata_anchor",
    "select_private_metadata_anchor_gid",
]
