# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Workspace geometry shared by TurboQuant runtime and KV planning.

Continuation prefill must not grow CUDA allocator state after the model runner
locks its workspace.  Keep the runtime tensor list and the bytes subtracted by
the KV planner in one small, dependency-light module so they cannot drift.
"""

from math import prod

import torch

WorkspaceSpec = tuple[tuple[int, ...], torch.dtype]


def continuation_prefill_workspace_specs(
    *,
    alloc_len: int,
    max_query_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    activation_dtype: torch.dtype,
    key_fp8: bool,
    prefix_combine: bool,
) -> dict[str, WorkspaceSpec]:
    """Return every tensor simultaneously live in one continuation call.

    Buffers use NHD layout so FP16 K8V4 dequant can write directly into the
    final FlashInfer K/V tensors.  Other TQ modes add only the conversion
    buffers that their dtype/rotation actually requires.
    """
    if alloc_len <= 0 or max_query_len <= 0:
        raise ValueError("alloc_len and max_query_len must be positive")

    kv_shape = (alloc_len, num_kv_heads, head_dim)
    out_shape = (max_query_len, num_q_heads, head_dim)
    lse_qh_shape = (max_query_len, num_q_heads)
    lse_hq_shape = (num_q_heads, max_query_len)

    specs: dict[str, WorkspaceSpec] = {
        "k_dequant": (kv_shape, torch.float16),
        "v_dequant": (kv_shape, torch.float16),
    }
    if not key_fp8:
        specs["k_rotated"] = (kv_shape, torch.float16)
    if activation_dtype != torch.float16:
        specs["k_full"] = (kv_shape, activation_dtype)
        specs["v_full"] = (kv_shape, activation_dtype)

    if prefix_combine:
        specs.update(
            {
                "prefix_out": (out_shape, activation_dtype),
                "current_out": (out_shape, activation_dtype),
                "merged_out": (out_shape, activation_dtype),
                "prefix_lse": (lse_qh_shape, torch.float32),
                "current_lse": (lse_qh_shape, torch.float32),
                "prefix_lse_hq": (lse_hq_shape, torch.float32),
                "current_lse_hq": (lse_hq_shape, torch.float32),
            }
        )
    else:
        specs["attn_out"] = (out_shape, activation_dtype)
    return specs


def workspace_specs_bytes(specs: dict[str, WorkspaceSpec]) -> int:
    """Match WorkspaceManager's per-view 256-byte alignment exactly."""
    total = 0
    for shape, dtype in specs.values():
        actual = prod(shape) * dtype.itemsize
        total += (actual + 255) // 256 * 256
    return total


def continuation_prefill_reservation_specs(**kwargs) -> dict[str, WorkspaceSpec]:
    """Return the larger of standard and prefix-combine live tensor sets."""
    standard = continuation_prefill_workspace_specs(
        **kwargs, prefix_combine=False
    )
    combined = continuation_prefill_workspace_specs(
        **kwargs, prefix_combine=True
    )
    if workspace_specs_bytes(combined) >= workspace_specs_bytes(standard):
        return combined
    return standard
