# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2 backport for the vLLM 0.21-era proposer stack.

This module ports the model-side pieces of upstream vLLM PR #52816 while
keeping the existing DFlash proposer architecture used by this repository.
"""

from collections.abc import Callable
from functools import cache
import os

import torch
import torch.nn.functional as F
from torch import nn

from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
    tensor_model_parallel_all_gather,
)
from vllm.logger import init_logger
from vllm.model_executor.layers.linear import ReplicatedLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    UnquantizedEmbeddingMethod,
)
from vllm.multimodal.inputs import NestedTensors
from vllm.platforms import current_platform
from vllm.utils.flashinfer import has_flashinfer

from .qwen3_dflash import (
    DFlashQwen3DecoderLayer,
    DFlashQwen3ForCausalLM,
    DFlashQwen3Model,
)
from .utils import maybe_prefix

logger = init_logger(__name__)


def _fp32_island_enabled() -> bool:
    return os.environ.get("VLLM_DFLASH_FP32_ISLAND") == "1"


def _fp32_rms_norm_add(
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    norm: nn.Module,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined = hidden_states.float()
    if residual is not None:
        combined = combined + residual.float()
    variance = combined.square().mean(dim=-1, keepdim=True)
    weight = norm.weight.float()
    normalized = combined * torch.rsqrt(variance + norm.variance_epsilon) * weight
    return normalized, combined

_flashinfer_topk_broken = False


@cache
def _flashinfer_topk() -> Callable[..., tuple[torch.Tensor, torch.Tensor]] | None:
    """Return FlashInfer radix top-k when the installed version exposes it."""
    if not current_platform.is_cuda() or not has_flashinfer():
        return None
    try:
        from flashinfer import top_k
    except (ImportError, AttributeError):
        logger.info_once(
            "FlashInfer top_k is unavailable; DFlash2 will use torch.topk."
        )
        return None
    return top_k


def _topk(scores: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Use FlashInfer when it works on this GPU, otherwise permanently fall back."""
    global _flashinfer_topk_broken
    if scores.is_cuda and torch.cuda.get_device_capability(scores.device)[0] < 8:
        # FlashInfer 0.6.8 can silently return corrupt indices on SM75 for the
        # large vocab/logit layout used by DFlash2. An exception-based fallback
        # cannot protect correctness, so keep all Turing paths on torch.topk.
        major, minor = torch.cuda.get_device_capability(scores.device)
        logger.warning_once(
            "DFlash2 using torch.topk on SM%d%d; FlashInfer top_k is disabled "
            "because this Turing path can return incorrect indices.",
            major,
            minor,
        )
        return torch.topk(scores, k, dim=-1)
    impl = None if _flashinfer_topk_broken else _flashinfer_topk()
    if impl is None or not scores.is_cuda:
        return torch.topk(scores, k, dim=-1)
    try:
        return impl(scores, k, sorted=True, deterministic=True)
    except Exception as exc:
        # FlashInfer supports SM75 in general, but individual kernels/version
        # combinations can still be unavailable. DFlash2 must remain usable on
        # Turing, so turn this into a one-time fallback instead of a startup failure.
        _flashinfer_topk_broken = True
        logger.warning_once(
            "FlashInfer top_k failed for DFlash2 (%s); falling back to torch.topk.",
            exc,
        )
        return torch.topk(scores, k, dim=-1)


def _grouped_conv(
    hidden_states: torch.Tensor,
    delta: torch.Tensor,
    base: torch.Tensor,
    block_size: int,
    num_groups: int,
    group_size: int,
    taps: int,
) -> torch.Tensor:
    blocks = hidden_states.unflatten(-1, (num_groups, group_size))
    coefficients = base.view(1, taps, num_groups, group_size) + delta.unsqueeze(-1)
    output = coefficients[:, 0] * blocks
    position = torch.arange(hidden_states.shape[0], device=hidden_states.device)
    if block_size & (block_size - 1) == 0:
        position = position & (block_size - 1)
    else:
        position = position % block_size
    for tap in range(1, taps):
        shifted = F.pad(blocks[:-tap], (0, 0, 0, 0, tap, 0))
        output += coefficients[:, tap] * shifted * (position >= tap).view(-1, 1, 1)
    return output.flatten(-2)


class DFlashGroupedConv(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        taps: int,
        group_size: int,
        block_size: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"conv_group_size={group_size} must divide hidden_size={hidden_size}."
            )
        self.block_size = block_size
        self.taps = taps
        self.group_size = group_size
        self.num_groups = hidden_size // group_size
        self.base_kernel = nn.Parameter(
            torch.empty(2, taps, hidden_size, dtype=params_dtype),
            requires_grad=False,
        )
        self.kernel_projection = ReplicatedLinear(
            hidden_size,
            2 * taps * self.num_groups,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "kernel_projection"),
            return_bias=False,
        )

    def _convolve(
        self, hidden_states: torch.Tensor, delta: torch.Tensor, side: int
    ) -> torch.Tensor:
        return _grouped_conv(
            hidden_states,
            delta,
            self.base_kernel[side],
            self.block_size,
            self.num_groups,
            self.group_size,
            self.taps,
        )

    def prepare(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = self.kernel_projection(hidden_states).reshape(
            hidden_states.shape[0], 2, self.taps, self.num_groups
        )
        return self._convolve(hidden_states, coefficients[:, 0], 0), coefficients[:, 1]

    def finish(
        self, hidden_states: torch.Tensor, coefficients: torch.Tensor
    ) -> torch.Tensor:
        if _fp32_island_enabled():
            return self._convolve(
                hidden_states.float(), coefficients.float(), 1
            )
        return self._convolve(hidden_states, coefficients, 1)


class DFlash2Qwen3DecoderLayer(nn.Module):
    """DFlash2 layer built by reusing the already-constructed DFlash1 modules.

    The fork's DFlash base model predates upstream's decoder_layer_cls hook.
    Re-wrapping preserves parameter names and tensors while adding only the two
    DFlash2 convolution modules, avoiding a second allocation of attention/MLP weights.
    """

    def __init__(
        self,
        base_layer: DFlashQwen3DecoderLayer,
        *,
        vllm_config: VllmConfig,
        prefix: str,
    ) -> None:
        super().__init__()
        self.hidden_size = base_layer.hidden_size
        self.self_attn = base_layer.self_attn
        self.mlp = base_layer.mlp
        self.input_layernorm = base_layer.input_layernorm
        self.post_attention_layernorm = base_layer.post_attention_layernorm

        config = vllm_config.speculative_config.draft_model_config.hf_config
        draft_config = config.dflash_config
        conv_args = dict(
            hidden_size=config.hidden_size,
            taps=int(draft_config["conv_kernel_size"]),
            group_size=int(draft_config["conv_group_size"]),
            block_size=1 + vllm_config.speculative_config.num_speculative_tokens,
            params_dtype=vllm_config.model_config.dtype,
        )
        self.attention_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "attention_conv")
        )
        self.mlp_conv = DFlashGroupedConv(
            **conv_args, prefix=maybe_prefix(prefix, "mlp_conv")
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if _fp32_island_enabled():
            hidden_states, residual = _fp32_rms_norm_add(
                hidden_states, residual, self.input_layernorm
            )
            hidden_states = hidden_states.to(self.input_layernorm.weight.dtype)
        else:
            if residual is None:
                residual = hidden_states
                hidden_states = self.input_layernorm(hidden_states)
            else:
                hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states, coefficients = self.attention_conv.prepare(hidden_states)
        hidden_states = self.self_attn(positions=positions, hidden_states=hidden_states)
        hidden_states = self.attention_conv.finish(hidden_states, coefficients)

        if _fp32_island_enabled():
            hidden_states, residual = _fp32_rms_norm_add(
                hidden_states, residual, self.post_attention_layernorm
            )
            hidden_states = hidden_states.to(self.post_attention_layernorm.weight.dtype)
        else:
            hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states, coefficients = self.mlp_conv.prepare(hidden_states)
        if _fp32_island_enabled():
            gate_up, _ = self.mlp.gate_up_proj(hidden_states)
            activated = self.mlp.act_fn(gate_up)
            local = F.linear(
                activated.float(), self.mlp.down_proj.weight.float(), bias=None
            )
            hidden_states = tensor_model_parallel_all_reduce(local)
        else:
            hidden_states = self.mlp(hidden_states)
        hidden_states = self.mlp_conv.finish(hidden_states, coefficients)
        return hidden_states, residual


def _score_edges(
    predecessor_table: torch.Tensor,
    successor_table: torch.Tensor,
    candidate_ids: torch.Tensor,
    unary_logits: torch.Tensor,
    hidden: torch.Tensor,
    anchor_token_ids: torch.Tensor,
    top_k: int,
) -> torch.Tensor:
    successors = successor_table[candidate_ids]
    predecessor_ids = torch.cat(
        (
            anchor_token_ids[:, None, None].expand(-1, 1, top_k),
            candidate_ids[:, :-1],
        ),
        dim=1,
    )
    predecessors = predecessor_table[predecessor_ids]
    return unary_logits[:, :, None] + torch.einsum(
        "blpr,blcr->blpc", predecessors * hidden[:, :, None], successors
    )


@support_torch_compile
class CandidateSelector(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        vocab_size: int,
        rank: int,
        top_k: int,
        params_dtype: torch.dtype,
        prefix: str,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.predecessor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.successor_codebook = nn.Parameter(
            torch.empty(vocab_size, rank, dtype=params_dtype), requires_grad=False
        )
        self.hidden_projection = ReplicatedLinear(
            hidden_size,
            rank,
            bias=False,
            params_dtype=params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "hidden_projection"),
            return_bias=False,
        )

    def forward(
        self,
        candidate_ids: torch.Tensor,
        unary_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        anchor_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.hidden_projection(hidden_states)
        return _score_edges(
            self.predecessor_codebook,
            self.successor_codebook,
            candidate_ids,
            unary_logits,
            hidden,
            anchor_token_ids,
            self.top_k,
        )


class DFlash2Qwen3Model(DFlashQwen3Model):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        start_layer_id: int = 0,
        prefix: str = "",
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            start_layer_id=start_layer_id,
            prefix=prefix,
        )
        draft_config = self.config.dflash_config
        self.input_embedding_scale = float(
            draft_config.get("input_embedding_scale", 1.0)
        )

        # Replace each base DFlash layer with a wrapper that reuses its existing
        # submodules and adds the two DFlash2 convolutions. Parameter names remain
        # model.layers.N.{self_attn,mlp,...}, matching the checkpoint.
        for local_idx, base_layer in enumerate(list(self.layers)):
            layer_prefix = maybe_prefix(
                prefix, f"layers.{local_idx + start_layer_id}"
            )
            self.layers[local_idx] = DFlash2Qwen3DecoderLayer(
                base_layer,
                vllm_config=vllm_config,
                prefix=layer_prefix,
            )

        with set_model_tag("dflash2_candidate_selector"):
            self.candidate_selector = CandidateSelector(
                hidden_size=self.config.hidden_size,
                vocab_size=self.config.vocab_size,
                rank=int(draft_config["selector_rank"]),
                top_k=int(draft_config["selector_top_k"]),
                params_dtype=vllm_config.model_config.dtype,
                prefix=maybe_prefix(prefix, "candidate_selector"),
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return super().embed_input_ids(input_ids) * self.input_embedding_scale

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        input_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not _fp32_island_enabled():
            return super().forward(input_ids, positions, input_embeds)
        if input_embeds is None:
            input_embeds = self.embed_input_ids(input_ids)
        hidden_states = input_embeds
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
        hidden_states, _ = _fp32_rms_norm_add(hidden_states, residual, self.norm)
        return hidden_states.to(input_embeds.dtype)


class DFlash2Qwen3ForCausalLM(DFlashQwen3ForCausalLM):
    """DFlash2 draft model adapted to this fork's pre-V2 proposer stack."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # Mirror DFlashQwen3ForCausalLM.__init__, but instantiate DFlash2Qwen3Model.
        nn.Module.__init__(self)
        speculative_config = vllm_config.speculative_config
        assert speculative_config is not None
        self.config = speculative_config.draft_model_config.hf_config
        if getattr(self.config, "draft_vocab_size", None) is None:
            self.config.draft_vocab_size = getattr(self.config, "vocab_size", None)

        target_layer_num = vllm_config.model_config.get_num_layers(
            vllm_config.parallel_config
        )
        self.config.target_layer_count = target_layer_num
        self.model = DFlash2Qwen3Model(
            vllm_config=vllm_config,
            prefix="model",
            start_layer_id=target_layer_num,
        )

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.lm_head = ParallelLMHead(
            self.config.draft_vocab_size,
            self.config.hidden_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        self.logits_processor = LogitsProcessor(
            self.config.draft_vocab_size, scale=logit_scale
        )

        target_vocab_size = vllm_config.model_config.get_vocab_size()
        if self.config.draft_vocab_size != target_vocab_size:
            self.draft_id_to_target_id = nn.Parameter(
                torch.zeros(self.config.draft_vocab_size, dtype=torch.long),
                requires_grad=False,
            )
        else:
            self.draft_id_to_target_id = None

        draft_config = self.config.dflash_config
        self.output_multiplier = float(draft_config.get("output_multiplier", 1.0))
        softcap = float(draft_config.get("final_logit_softcapping") or 0.0)
        self.final_logit_softcapping = softcap if softcap > 0 else None

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: NestedTensors | None = None,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def compute_candidates(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.draft_id_to_target_id is not None:
            raise ValueError(
                "DFlash2 candidate selection does not support draft-vocabulary "
                "remapping; the draft and target vocabularies must match."
            )

        if not isinstance(
            self.lm_head.quant_method,
            (UnquantizedEmbeddingMethod, UnquantizedLinearMethod),
        ):
            raise ValueError(
                "DFlash2 requires an unquantized target LM head for candidate TopK; "
                f"got {type(self.lm_head.quant_method).__name__}."
            )

        selector = self.model.candidate_selector
        logits = self.lm_head.quant_method.apply(self.lm_head, hidden_states, bias=None)
        num_pad = self.lm_head.shard_indices.num_org_vocab_padding
        if num_pad > 0:
            logits[..., -num_pad:] = -float("inf")
        values, ids = _topk(logits, selector.top_k)
        ids = ids.to(torch.int64) + self.lm_head.shard_indices.org_vocab_start_index

        if get_tensor_model_parallel_world_size() > 1:
            values = tensor_model_parallel_all_gather(values, dim=-1)
            ids = tensor_model_parallel_all_gather(ids, dim=-1)
            values, selected = _topk(values, selector.top_k)
            ids = ids.gather(-1, selected)

        values = values.float() * self.output_multiplier
        if self.final_logit_softcapping is not None:
            cap = self.final_logit_softcapping
            values = torch.tanh(values / cap) * cap
        return ids, values


EntryClass = DFlash2Qwen3ForCausalLM
