# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental private/windowed KV cache for DFlash2.

The legacy vLLM hybrid KV manager has to co-plan the target model's quantized
KV/state caches and DFlash's native FP16 attention cache.  On Qwen3.5/3.8 this
causes the five DFlash attention layers to change the target's otherwise compact
hybrid grouping and, on the compatibility path, to inflate smaller physical
pages to the largest DFlash page.  That is acceptable as a correctness bridge
but destroys long-context capacity.

This module provides an opt-in escape hatch for the validated batch=1 DFlash2
path.  When ``VLLM_DFLASH_PRIVATE_KV_WINDOW`` is a positive integer:

* EngineCore removes only the DFlash draft attention layers from the centrally
  managed KV specs, leaving the target's native KV planner untouched.
* Workers allocate one fixed-size FP16 ring cache for the DFlash layers during
  model loading, before target KV-memory profiling.
* Target hidden-state K/V for the most recent window are written to that ring.
* DFlash query attention is computed directly from the private context K/V plus
  the current bonus/mask query K/V.  The target verifier still attends the full
  target context, so truncating the *draft* context can only affect acceptance,
  not the verifier's correctness contract.

The first implementation is intentionally batch=1 and draft-eager.  It is an
experimental long-context path until acceptance/throughput are measured on the
real SM75 host.
"""

from __future__ import annotations

import os
import re
from functools import cache, wraps
from typing import Any

import torch
import torch.nn.functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)

_PRIVATE_WORKER_INSTALLED = False
_DRAFT_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.self_attn\.attn$")


@cache
def get_dflash_private_kv_window() -> int:
    """Return the opt-in DFlash private KV window in tokens (0 = disabled)."""
    raw = os.environ.get("VLLM_DFLASH_PRIVATE_KV_WINDOW", "0").strip().lower()
    if raw in {"", "0", "off", "false", "no", "disabled"}:
        return 0
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            "VLLM_DFLASH_PRIVATE_KV_WINDOW must be 0/off or a positive token count"
        ) from exc
    if value <= 0:
        raise ValueError("VLLM_DFLASH_PRIVATE_KV_WINDOW must be positive when enabled")
    return value


def dflash_private_kv_enabled() -> bool:
    return get_dflash_private_kv_window() > 0


def _draft_layer_bounds(vllm_config: Any) -> tuple[int, int]:
    target_layers = int(
        vllm_config.model_config.get_num_layers(vllm_config.parallel_config)
    )
    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_hf = speculative_config.draft_model_config.hf_config
    draft_layers = int(getattr(draft_hf, "num_hidden_layers", 0) or 0)
    if draft_layers <= 0:
        raise RuntimeError("DFlash private KV could not determine draft layer count")
    return target_layers, target_layers + draft_layers


def _is_dflash_draft_attn_layer(
    layer_name: str, target_start: int, draft_end: int
) -> bool:
    match = _DRAFT_LAYER_RE.search(layer_name)
    if match is None:
        return False
    layer_idx = int(match.group(1))
    return target_start <= layer_idx < draft_end


def filter_dflash2_private_kv_specs(
    vllm_config: Any,
    kv_cache_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove DFlash attention specs from EngineCore planning in private mode.

    Worker RPC still reports the draft Attention modules.  Filtering centrally
    avoids depending on model-construction/monkeypatch ordering and restores the
    target model's exact native KV grouping inputs.
    """
    window = get_dflash_private_kv_window()
    if window <= 0:
        return kv_cache_specs

    target_start, draft_end = _draft_layer_bounds(vllm_config)
    filtered: list[dict[str, Any]] = []
    removed_names: set[str] = set()
    removed_real_bytes = 0
    removed_padded_bytes = 0

    for worker_specs in kv_cache_specs:
        worker_filtered: dict[str, Any] = {}
        for name, spec in worker_specs.items():
            if _is_dflash_draft_attn_layer(name, target_start, draft_end):
                removed_names.add(name)
                removed_padded_bytes += int(getattr(spec, "page_size_bytes", 0) or 0)
                removed_real_bytes += int(
                    getattr(spec, "real_page_size_bytes", 0)
                    or getattr(spec, "page_size_bytes", 0)
                    or 0
                )
                continue
            worker_filtered[name] = spec
        filtered.append(worker_filtered)

    expected = draft_end - target_start
    if len(removed_names) < expected:
        raise RuntimeError(
            "DFlash private KV is enabled but EngineCore could not identify all "
            f"draft attention specs: expected at least {expected}, found "
            f"{len(removed_names)} ({sorted(removed_names)}). Refusing to fall "
            "back to the capacity-destroying managed DFlash KV layout."
        )

    logger.info_once(
        "DFlash2 private KV window=%d removed %d draft attention layers from "
        "EngineCore KV planning (reported real pages %.2f MiB, planned pages "
        "%.2f MiB per worker-spec set before filtering). Target KV specs remain "
        "unchanged.",
        window,
        len(removed_names),
        removed_real_bytes / 2**20,
        removed_padded_bytes / 2**20,
    )
    return filtered


def _advance_window_state(
    previous_start: int,
    previous_last: int,
    first_pos: int,
    last_pos: int,
    window: int,
) -> tuple[int, int]:
    """Track a contiguous suffix without retaining stale data across requests.

    Small overlaps are allowed because rejection/replay can revisit a handful of
    recent positions.  A position-0 chunk, a forward gap, or a large backwards
    jump starts a new private-cache epoch.  Prefix-cache hits that start at a
    non-zero position are therefore represented by the suffix actually observed
    by DFlash rather than by stale data from a previous request.
    """
    if first_pos < 0 or last_pos < first_pos:
        raise ValueError(f"invalid DFlash context positions {first_pos}..{last_pos}")
    same_stream = (
        previous_last >= 0
        and first_pos != 0
        and first_pos <= previous_last + 1
        and first_pos >= previous_last - 64
    )
    if same_stream:
        start = min(previous_start, first_pos)
        end = max(previous_last, last_pos)
    else:
        start = first_pos
        end = last_pos
    start = max(start, end - window + 1)
    return start, end


def install_dflash2_private_kv_worker_compat() -> bool:
    """Patch only DFlash worker/model paths when private KV is explicitly enabled."""
    global _PRIVATE_WORKER_INSTALLED
    window = get_dflash_private_kv_window()
    if window <= 0:
        return False
    if _PRIVATE_WORKER_INSTALLED:
        return True

    # Imports are intentionally lazy so EngineCore can use the spec-filter helper
    # without importing CUDA/model-runner modules.
    from vllm import _custom_ops as ops
    from vllm.v1.attention.backend import CommonAttentionMetadata
    from vllm.v1.attention.backends.utils import PAD_SLOT_ID
    from vllm.v1.config import CUDAGraphMode if False else None  # type: ignore
    from vllm.config import CUDAGraphMode
    from vllm.model_executor.models import qwen3_dflash as dflash_model
    from vllm.v1.spec_decode.dflash import DFlashProposer

    # ------------------------------------------------------------------
    # DFlash attention: bypass managed Attention/FlexAttention and read the
    # private fixed-window cache directly.
    # ------------------------------------------------------------------
    attn_cls = dflash_model.DFlashQwen3Attention
    original_attn_init = attn_cls.__init__
    original_attn_forward = attn_cls.forward

    if not getattr(original_attn_forward, "_dflash_private_kv", False):

        @wraps(original_attn_init)
        def private_attn_init(self, *args, **kwargs):
            original_attn_init(self, *args, **kwargs)
            self._dflash_private_kv_window = window
            self._dflash_private_cache = None
            self._dflash_private_state = None

        @wraps(original_attn_forward)
        def private_attn_forward(
            self,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
        ) -> torch.Tensor:
            if getattr(self, "_dflash_private_kv_window", 0) <= 0:
                return original_attn_forward(self, positions, hidden_states)

            qkv = F.linear(hidden_states, self.qkv_proj.weight, self.qkv_proj.bias)
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

            q_shape, k_shape = q.shape, k.shape
            q = self.q_norm(
                q.view(*q_shape[:-1], q_shape[-1] // self.head_dim, self.head_dim)
            ).view(q_shape)
            k = self.k_norm(
                k.view(*k_shape[:-1], k_shape[-1] // self.head_dim, self.head_dim)
            ).view(k_shape)
            q, k = self.rotary_emb(positions, q, k)

            q = q.view(-1, self.num_heads, self.head_dim)
            k = k.view(-1, self.num_kv_heads, self.head_dim)
            v = v.view(-1, self.num_kv_heads, self.head_dim)

            cache = self._dflash_private_cache
            state = self._dflash_private_state
            ctx_len = int(state.get("valid_len", 0)) if state is not None else 0
            if cache is not None and ctx_len > 0:
                if ctx_len == window or bool(state.get("direct_prefix", False)):
                    ctx_k = cache[0, :ctx_len]
                    ctx_v = cache[1, :ctx_len]
                else:
                    slots = state["slot_indices"][:ctx_len]
                    ctx_k = cache[0].index_select(0, slots)
                    ctx_v = cache[1].index_select(0, slots)
                all_k = torch.cat((ctx_k, k), dim=0)
                all_v = torch.cat((ctx_v, v), dim=0)
            else:
                all_k = k
                all_v = v

            # [1, heads, query, dim] x [1, kv_heads, context, dim].  RoPE is
            # already applied to Q/K, and DFlash uses non-causal attention for
            # its bonus+mask query lattice, so physical ring order is irrelevant.
            q_sdpa = q.transpose(0, 1).unsqueeze(0)
            k_sdpa = all_k.transpose(0, 1).unsqueeze(0)
            v_sdpa = all_v.transpose(0, 1).unsqueeze(0)
            attn_output = F.scaled_dot_product_attention(
                q_sdpa,
                k_sdpa,
                v_sdpa,
                dropout_p=0.0,
                is_causal=False,
                scale=self.scaling,
                enable_gqa=(self.num_heads != self.num_kv_heads),
            )
            attn_output = attn_output.squeeze(0).transpose(0, 1).reshape(
                -1, self.q_size
            )
            output, _ = self.o_proj(attn_output)
            return output

        private_attn_forward._dflash_private_kv = True  # type: ignore[attr-defined]
        attn_cls.__init__ = private_attn_init
        attn_cls.forward = private_attn_forward

    # ------------------------------------------------------------------
    # DFlash model: allocate private KV before memory profiling and write only
    # the recent window during fused context precomputation.
    # ------------------------------------------------------------------
    model_cls = dflash_model.DFlashQwen3Model
    original_build = model_cls._build_fused_kv_buffers
    original_precompute = model_cls.precompute_and_store_context_kv

    if not getattr(original_build, "_dflash_private_kv", False):

        @wraps(original_build)
        def build_fused_with_private_kv(self):
            original_build(self)
            if getattr(self, "_dflash_private_kv_cache", None) is not None:
                return

            dtype = self._fused_kv_weight.dtype
            device = self._fused_kv_weight.device
            cache = torch.empty(
                (
                    self._num_attn_layers,
                    2,
                    window,
                    self._num_kv_heads,
                    self._head_dim,
                ),
                dtype=dtype,
                device=device,
            )
            state: dict[str, Any] = {
                "window": window,
                "start_pos": 0,
                "last_pos": -1,
                "valid_len": 0,
                "direct_prefix": True,
                "arange": torch.arange(window, dtype=torch.long, device=device),
                "slot_indices": torch.empty(window, dtype=torch.long, device=device),
            }
            self._dflash_private_kv_cache = cache
            self._dflash_private_kv_state = state
            for layer_idx, layer in enumerate(self.layers):
                outer_attn = layer.self_attn
                outer_attn._dflash_private_cache = cache[layer_idx]
                outer_attn._dflash_private_state = state

            logger.info_once(
                "Allocated DFlash2 private KV window=%d for %d layers: %.2f MiB "
                "per rank, dtype=%s. Allocation occurs during model loading so "
                "target KV sizing accounts for it.",
                window,
                self._num_attn_layers,
                cache.numel() * cache.element_size() / 2**20,
                dtype,
            )

        @wraps(original_precompute)
        def precompute_private_kv(
            self,
            context_states: torch.Tensor,
            context_positions: torch.Tensor,
            context_slot_mapping: torch.Tensor | None = None,
        ) -> None:
            if not hasattr(self, "_num_attn_layers"):
                self._build_fused_kv_buffers()
            if getattr(self, "_dflash_private_kv_cache", None) is None:
                return original_precompute(
                    self, context_states, context_positions, context_slot_mapping
                )

            # A single target chunk can exceed the entire draft window.  Do not
            # project K/V that can never be observed by the draft.
            if context_states.shape[0] > window:
                context_states = context_states[-window:]
                context_positions = context_positions[-window:]
                if context_slot_mapping is not None:
                    context_slot_mapping = context_slot_mapping[-window:]

            num_ctx = context_states.shape[0]
            if num_ctx <= 0:
                return
            L = self._num_attn_layers
            kv = self._kv_size
            hd = self._head_dim
            nkv = self._num_kv_heads

            normed_context_states = torch.empty_like(context_states)
            ops.rms_norm(
                normed_context_states,
                context_states,
                self._hidden_norm_weight,
                self._rms_norm_eps,
            )
            all_kv_flat = F.linear(
                normed_context_states, self._fused_kv_weight, self._fused_kv_bias
            )
            all_kv = (
                all_kv_flat.view(num_ctx, L, 2, nkv, hd)
                .permute(2, 1, 0, 3, 4)
                .contiguous()
            )
            all_k = all_kv[0]
            all_v = all_kv[1]
            all_k_normed = torch.empty_like(all_k)
            for i in range(L):
                ops.rms_norm(
                    all_k_normed[i],
                    all_k[i],
                    self._k_norm_weights[i],
                    self._rms_norm_eps,
                )

            all_k_flat = all_k_normed.view(L * num_ctx, kv)
            positions_repeated = context_positions.repeat(L)
            cos_sin_cache = self._rope_cos_sin_cache
            if cos_sin_cache.dtype != all_k_flat.dtype:
                cos_sin_cache = cos_sin_cache.to(dtype=all_k_flat.dtype)
            ops.rotary_embedding(
                positions_repeated,
                all_k_flat,
                None,
                self._rope_head_size,
                cos_sin_cache,
                self._rope_is_neox,
            )

            # Dummy/profile calls intentionally do not mutate the persistent ring.
            if context_slot_mapping is None:
                return

            positions_i64 = context_positions.to(torch.long)
            slots = positions_i64.remainder(window)
            all_k_final = all_k_flat.view(L, num_ctx, nkv, hd)
            cache = self._dflash_private_kv_cache
            cache[:, 0].index_copy_(1, slots, all_k_final)
            cache[:, 1].index_copy_(1, slots, all_v)

            # One small host read per proposal keeps the five layer forwards free
            # of dynamic GPU reductions.  The draft is eager on the validated SM75
            # path; this is intentionally correctness/long-context first.
            first_pos = int(context_positions[0].item())
            last_pos = int(context_positions[-1].item())
            state = self._dflash_private_kv_state
            start_pos, end_pos = _advance_window_state(
                int(state["start_pos"]),
                int(state["last_pos"]),
                first_pos,
                last_pos,
                window,
            )
            valid_len = min(window, end_pos - start_pos + 1)
            state["start_pos"] = start_pos
            state["last_pos"] = end_pos
            state["valid_len"] = valid_len
            state["direct_prefix"] = start_pos == 0 or valid_len == window
            if valid_len < window and start_pos != 0:
                slot_view = state["slot_indices"][:valid_len]
                torch.add(state["arange"][:valid_len], start_pos, out=slot_view)
                slot_view.remainder_(window)

            logger.debug(
                "DFlash2 private KV updated context=%d..%d retained=%d..%d (%d tokens)",
                first_pos,
                last_pos,
                start_pos,
                end_pos,
                valid_len,
            )

        build_fused_with_private_kv._dflash_private_kv = True  # type: ignore[attr-defined]
        precompute_private_kv._dflash_private_kv = True  # type: ignore[attr-defined]
        model_cls._build_fused_kv_buffers = build_fused_with_private_kv
        model_cls.precompute_and_store_context_kv = precompute_private_kv

    # ------------------------------------------------------------------
    # Legacy proposer: the draft no longer owns an EngineCore KV group.
    # Prepare only bonus/mask tokens + positions, and keep all draft metadata
    # empty because private attention does not call vLLM Attention custom ops.
    # ------------------------------------------------------------------
    proposer_cls = DFlashProposer
    original_load_model = proposer_cls.load_model
    original_initialize_attn_backend = proposer_cls.initialize_attn_backend
    original_set_inputs = proposer_cls.set_inputs_first_pass
    original_initialize_cg = proposer_cls.initialize_cudagraph_keys

    if not getattr(original_load_model, "_dflash_private_kv", False):

        @wraps(original_load_model)
        def load_model_private(self, target_model):
            original_load_model(self, target_model)
            if getattr(self, "_is_dflash2", False):
                removed = len(self._draft_attn_layer_names)
                self._draft_attn_layer_names = set()
                logger.info_once(
                    "DFlash2 private KV detached %d draft attention layers from "
                    "the managed KV metadata path.",
                    removed,
                )

        @wraps(original_initialize_attn_backend)
        def initialize_attn_backend_private(
            self,
            kv_cache_config,
            kernel_block_sizes=None,
        ):
            if getattr(self, "_is_dflash2", False):
                self.draft_attn_groups = []
                self.kv_cache_gid = -1
                self.block_size = int(self.vllm_config.cache_config.block_size or 1)
                return
            return original_initialize_attn_backend(
                self, kv_cache_config, kernel_block_sizes
            )

        @wraps(original_set_inputs)
        def set_inputs_private(
            self,
            target_token_ids: torch.Tensor,
            next_token_ids: torch.Tensor,
            target_positions: torch.Tensor,
            target_hidden_states: torch.Tensor,
            token_indices_to_sample: torch.Tensor | None,
            cad: CommonAttentionMetadata,
            num_rejected_tokens_gpu: torch.Tensor | None,
        ):
            if not getattr(self, "_is_dflash2", False):
                return original_set_inputs(
                    self,
                    target_token_ids,
                    next_token_ids,
                    target_positions,
                    target_hidden_states,
                    token_indices_to_sample,
                    cad,
                    num_rejected_tokens_gpu,
                )
            if cad.batch_size() != 1:
                raise RuntimeError(
                    "DFlash2 private KV currently supports max_num_seqs=1 only"
                )

            num_context = int(target_token_ids.shape[0])
            num_rejected = (
                int(num_rejected_tokens_gpu[0].item())
                if num_rejected_tokens_gpu is not None
                else 0
            )
            valid_context = num_context - num_rejected
            if valid_context <= 0:
                raise RuntimeError("DFlash2 private KV received no valid context tokens")

            num_query_per_req = 1 + self.num_speculative_tokens
            self._dflash_num_context = valid_context
            self._dflash_hidden_states = target_hidden_states[:valid_context]
            self._context_positions_buffer[:valid_context].copy_(
                target_positions[:valid_context]
            )

            last_pos = target_positions[valid_context - 1]
            self.positions[:num_query_per_req].copy_(
                last_pos + 1 + self.arange[:num_query_per_req]
            )
            self.input_ids[0].copy_(next_token_ids[0])
            if num_query_per_req > 1:
                self.input_ids[1:num_query_per_req].fill_(
                    self.parallel_drafting_token_id
                )
            token_indices = self._token_indices_to_sample_buffer[
                : self.num_speculative_tokens
            ]
            token_indices.copy_(self.arange[1:num_query_per_req])

            # Private attention ignores managed slot mappings. Keep explicit PAD
            # values so accidental use fails closed instead of corrupting target KV.
            self._context_slot_mapping_buffer[:valid_context].fill_(PAD_SLOT_ID)
            self._slot_mapping_buffer[:num_query_per_req].fill_(PAD_SLOT_ID)

            effective_seq_lens = cad.seq_lens
            if num_rejected_tokens_gpu is not None:
                effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu
            upper = (
                cad.seq_lens_cpu_upper_bound + num_query_per_req
                if cad.seq_lens_cpu_upper_bound is not None
                else None
            )
            query_start_cpu = torch.tensor([0, num_query_per_req], dtype=torch.int32)
            new_cad = CommonAttentionMetadata(
                query_start_loc=self.arange[:2] * num_query_per_req,
                seq_lens=effective_seq_lens + num_query_per_req,
                query_start_loc_cpu=query_start_cpu,
                _seq_lens_cpu=None,
                _num_computed_tokens_cpu=None,
                seq_lens_cpu_upper_bound=upper,
                num_reqs=1,
                num_actual_tokens=num_query_per_req,
                max_query_len=num_query_per_req,
                max_seq_len=cad.max_seq_len + num_query_per_req,
                block_table_tensor=cad.block_table_tensor,
                slot_mapping=self._slot_mapping_buffer[:num_query_per_req],
                causal=False,
            )
            return num_query_per_req, token_indices, new_cad

        @wraps(original_initialize_cg)
        def initialize_cudagraph_keys_private(self, cudagraph_mode):
            if getattr(self, "_is_dflash2", False):
                self.cudagraph_dispatcher.initialize_cudagraph_keys(
                    CUDAGraphMode.NONE
                )
                logger.info_once(
                    "DFlash2 private KV forces the draft model eager; target CUDA "
                    "Graph remains independently enabled."
                )
                return
            return original_initialize_cg(self, cudagraph_mode)

        load_model_private._dflash_private_kv = True  # type: ignore[attr-defined]
        proposer_cls.load_model = load_model_private
        proposer_cls.initialize_attn_backend = initialize_attn_backend_private
        proposer_cls.set_inputs_first_pass = set_inputs_private
        proposer_cls.initialize_cudagraph_keys = initialize_cudagraph_keys_private

    _PRIVATE_WORKER_INSTALLED = True
    logger.info_once(
        "Installed experimental DFlash2 private KV worker path with window=%d tokens.",
        window,
    )
    return True


__all__ = [
    "dflash_private_kv_enabled",
    "filter_dflash2_private_kv_specs",
    "get_dflash_private_kv_window",
    "install_dflash2_private_kv_worker_compat",
]
