# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from .interfaces import (
    HasInnerState,
    SupportsLoRA,
    SupportsMRoPE,
    SupportsMultiModal,
    SupportsPP,
    SupportsTranscription,
    has_inner_state,
    supports_lora,
    supports_mrope,
    supports_multimodal,
    supports_pp,
    supports_transcription,
)
from .interfaces_base import (
    VllmModelForPooling,
    VllmModelForTextGeneration,
    is_pooling_model,
    is_text_generation_model,
)
from .registry import ModelRegistry

# Fork-local compatibility registration. Upstream vLLM PR #52816 adds this
# entry directly to the large static registry; registering it lazily here keeps
# this backport small while preserving the same public architecture name.
ModelRegistry.register_model(
    "DFlash2DraftModel",
    "vllm.model_executor.models.qwen3_dflash2:DFlash2Qwen3ForCausalLM",
)

__all__ = [
    "ModelRegistry",
    "VllmModelForPooling",
    "is_pooling_model",
    "VllmModelForTextGeneration",
    "is_text_generation_model",
    "HasInnerState",
    "has_inner_state",
    "SupportsLoRA",
    "supports_lora",
    "SupportsMultiModal",
    "supports_multimodal",
    "SupportsMRoPE",
    "supports_mrope",
    "SupportsPP",
    "supports_pp",
    "SupportsTranscription",
    "supports_transcription",
]
