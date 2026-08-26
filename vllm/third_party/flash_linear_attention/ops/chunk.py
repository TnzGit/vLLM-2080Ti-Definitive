# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

from math import prod

import torch

from .chunk_delta_h import chunk_gated_delta_rule_fwd_h
from .chunk_o import chunk_fwd_o
from .chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from .cumsum import chunk_local_cumsum
from .l2norm import l2norm_fwd
from .solve_tril import solve_tril
from .utils import FLA_CHUNK_SIZE, SUPPRESS_LEVEL, input_guard
from .wy_fast import recompute_w_u_fwd


def _workspace_manager():
    # Keep the bundled FLA module importable outside the V1 worker.  Importing
    # WorkspaceManager eagerly here creates a model-loader/worker import cycle.
    from vllm.v1.worker.workspace import (
        current_workspace_manager,
        is_workspace_manager_initialized,
    )

    if not is_workspace_manager_initialized():
        return None
    return current_workspace_manager()


def _gdn_prefill_workspace_specs(
    *,
    batch_size: int,
    num_tokens: int,
    num_heads: int,
    key_dim: int,
    value_dim: int,
    dtype: torch.dtype,
    num_chunks: int,
    num_sequences: int,
) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
    """Return every simultaneously-live FLA prefill scratch allocation.

    Keeping this list in one place is important: WorkspaceManager can reserve
    the complete peak before CUDA graph locking and serve compact typed views
    at inference time.  No tensor in this list escapes the GDN prefill call.
    """
    bt = FLA_CHUNK_SIZE
    return (
        ((batch_size, num_tokens, num_heads), torch.float32),  # cumulative g
        ((batch_size, num_tokens, num_heads, bt), torch.float32),  # raw A
        ((batch_size, num_tokens, num_heads, bt), dtype),  # solved A
        ((batch_size, num_tokens, num_heads, key_dim), dtype),  # w
        ((batch_size, num_tokens, num_heads, value_dim), dtype),  # u
        (
            (batch_size, num_chunks, num_heads, value_dim, key_dim),
            dtype,
        ),  # h
        ((batch_size, num_tokens, num_heads, value_dim), dtype),  # v_new
        ((num_sequences, num_heads, value_dim, key_dim), torch.float32),
    )


def _gdn_post_conv_workspace_specs(
    *,
    num_tokens: int,
    num_key_heads: int,
    num_value_heads: int,
    key_dim: int,
    value_dim: int,
    dtype: torch.dtype,
) -> tuple[tuple[tuple[int, ...], torch.dtype], ...]:
    """Return the five post-conv outputs that stay live during FLA prefill."""
    return (
        ((num_tokens, num_key_heads, key_dim), dtype),  # q
        ((num_tokens, num_key_heads, key_dim), dtype),  # k
        ((num_tokens, num_value_heads, value_dim), dtype),  # v
        ((num_tokens, num_value_heads), torch.float32),  # g
        ((num_tokens, num_value_heads), torch.float32),  # beta
    )


def get_gdn_post_conv_workspace(
    *,
    num_tokens: int,
    num_key_heads: int,
    num_value_heads: int,
    key_dim: int,
    value_dim: int,
    dtype: torch.dtype,
) -> list[torch.Tensor] | None:
    """Get stable output views for fused post-conv preparation.

    The subsequent chunk call requests the same five specs first and places
    its scratch after them.  This keeps q/k/v/g/beta live without aliasing the
    recurrent scratch, while both stages share one fixed backing allocation.
    """
    manager = _workspace_manager()
    if manager is None:
        return None
    return manager.get_simultaneous(
        *_gdn_post_conv_workspace_specs(
            num_tokens=num_tokens,
            num_key_heads=num_key_heads,
            num_value_heads=num_value_heads,
            key_dim=key_dim,
            value_dim=value_dim,
            dtype=dtype,
        )
    )


def reserve_gdn_prefill_workspace(
    *,
    max_num_batched_tokens: int,
    max_num_sequences: int,
    num_heads: int,
    num_key_heads: int,
    key_dim: int,
    value_dim: int,
    dtype: torch.dtype,
) -> int:
    """Reserve the worst-case flattened varlen GDN prefill workspace.

    A flattened batch can need one partial chunk per sequence.  The extra
    ``max_num_sequences - 1`` chunks make the reservation safe even when the
    scheduler splits the token budget across many short requests.
    """
    manager = _workspace_manager()
    if manager is None:
        return 0
    num_chunks = (
        (max_num_batched_tokens + FLA_CHUNK_SIZE - 1) // FLA_CHUNK_SIZE
        + max(0, max_num_sequences - 1)
    )
    post_conv_specs = _gdn_post_conv_workspace_specs(
        num_tokens=max_num_batched_tokens,
        num_key_heads=num_key_heads,
        num_value_heads=num_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        dtype=dtype,
    )
    scratch_specs = _gdn_prefill_workspace_specs(
        batch_size=1,
        num_tokens=max_num_batched_tokens,
        num_heads=num_heads,
        key_dim=key_dim,
        value_dim=value_dim,
        dtype=dtype,
        num_chunks=num_chunks,
        num_sequences=max_num_sequences,
    )
    specs = post_conv_specs + scratch_specs
    manager.get_simultaneous(*specs)
    return sum(prod(shape) * dt.itemsize for shape, dt in specs)


def _get_gdn_prefill_workspace(
    *,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None,
) -> list[torch.Tensor] | None:
    manager = _workspace_manager()
    if manager is None:
        return None
    B, T, _Hg, K = k.shape
    Hg = k.shape[-2]
    H = beta.shape[-1]
    V = v.shape[-1]
    if cu_seqlens is None:
        num_sequences = B
        num_chunks = (T + FLA_CHUNK_SIZE - 1) // FLA_CHUNK_SIZE
    else:
        num_sequences = len(cu_seqlens) - 1
        if chunk_indices is not None:
            num_chunks = len(chunk_indices)
        else:
            # prepare_chunk_indices uses one ceil-divided chunk group per seq.
            lengths = cu_seqlens[1:] - cu_seqlens[:-1]
            num_chunks = int(
                torch.div(
                    lengths + FLA_CHUNK_SIZE - 1,
                    FLA_CHUNK_SIZE,
                    rounding_mode="floor",
                )
                .sum()
                .item()
            )
    post_conv_specs = _gdn_post_conv_workspace_specs(
        num_tokens=T,
        num_key_heads=Hg,
        num_value_heads=H,
        key_dim=K,
        value_dim=V,
        dtype=k.dtype,
    )
    scratch_specs = list(
        _gdn_prefill_workspace_specs(
            batch_size=B,
            num_tokens=T,
            num_heads=H,
            key_dim=K,
            value_dim=V,
            dtype=k.dtype,
            num_chunks=num_chunks,
            num_sequences=num_sequences,
        )
    )
    if not output_final_state:
        scratch_specs.pop()
    outputs = manager.get_simultaneous(*post_conv_specs, *scratch_specs)
    return outputs[len(post_conv_specs) :]


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    core_attn_out: torch.Tensor | None = None,
):
    # This bundled FLA implementation is inference-only.  Use the shared
    # workspace for every inference layout, including speculative/mixed
    # batches whose final output cannot yet alias the model-runner buffer.
    scratch = _get_gdn_prefill_workspace(
        k=k,
        v=v,
        beta=beta,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    if scratch is None:
        g_out = A_out = Ai_out = w_out = u_out = h_out = v_new_out = None
        final_state_out = None
    else:
        g_out, A_out, Ai_out, w_out, u_out, h_out, v_new_out, *rest = scratch
        final_state_out = rest[0] if rest else None

    g = chunk_local_cumsum(
        g,
        chunk_size=FLA_CHUNK_SIZE,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        out=g_out,
    )
    # obtain WY representation. u is actually the new v.
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        beta=beta,
        g=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        output_dtype=torch.float32,
        out=A_out,
    )
    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        output_dtype=k.dtype,
        out=Ai_out,
    )
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g_cumsum=g,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        w_out=w_out,
        u_out=u_out,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        h_out=h_out,
        v_new_out=v_new_out,
        final_state_out=final_state_out,
    )
    o = chunk_fwd_o(
        q=q,
        k=k,
        v=v_new,
        h=h,
        g=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        core_attn_out=core_attn_out,
    )
    if SUPPRESS_LEVEL < 3:
        return g, o, A, final_state, None, None, None
    elif SUPPRESS_LEVEL >= 3:
        return g, o, A, final_state, w, h, v_new


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):
    @staticmethod
    @input_guard
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.Tensor | None = None,
        chunk_indices: torch.Tensor | None = None,
        chunk_offsets: torch.Tensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        core_attn_out: torch.Tensor | None = None,
    ):
        if use_qk_l2norm_in_kernel:
            q = l2norm_fwd(q)
            k = l2norm_fwd(k)

        g, o, A, final_state, w, h, v_new = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            core_attn_out=core_attn_out,
        )
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        if core_attn_out is not None:
            assert not torch.is_grad_enabled(), (
                "core_attn_out buffer reuse is only supported for inference"
            )
            assert q.dtype == o.dtype, "Incompatible dtype for inplace computation"
        return o.to(q.dtype), final_state


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    core_attn_out: torch.Tensor | None = None,
):
    r"""
    Args:
        q (torch.Tensor):
            Queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            Keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            Values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) Gating tensor (in log space!) of shape `[B, T, H]`.
        beta (torch.Tensor):
            Betas of shape `[B, T, H]`.
        scale (Optional[int]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, V, K]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, V, K]`. Default: `False`.
        cu_seqlens (torch.Tensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, V, K]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, V, K, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.int32)
        >>> o_var, ht_var = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    assert q.dtype == k.dtype == v.dtype
    assert q.dtype != torch.float32, (
        "ChunkGatedDeltaRuleFunction does not support float32. Please use bfloat16."
    )
    assert len(beta.shape) == 3, "beta must be of shape [B, T, H]."
    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing."
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}."
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        chunk_indices,
        chunk_offsets,
        use_qk_l2norm_in_kernel,
        core_attn_out,
    )
    return o, final_state
