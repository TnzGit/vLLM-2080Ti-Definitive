# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2 candidate-path selection for the legacy DFlash proposer."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _selector_walk_greedy_kernel(
    scores_ptr,
    candidate_ptr,
    tokens_ptr,
    num_steps: tl.constexpr,
    top_k: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Walk a DFlash2 candidate lattice with greedy edge selection.

    One Triton program owns one request, so there is no persistent request-index
    mapping to become stale when batches are reordered. This deliberately avoids
    the concurrency failure mode found while upstream PR #52816 was under review.
    """
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < top_k
    previous = 0

    for step in range(num_steps):
        flat = row * num_steps + step
        score_base = (flat * top_k + previous) * top_k
        scores = tl.load(
            scores_ptr + score_base + offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)

        # NaNs must never win the walk. +inf remains a valid strongest score and
        # an all--inf row deterministically falls back to the first candidate.
        scores = tl.where(scores == scores, scores, float("-inf"))
        _, index = tl.max(scores, axis=0, return_indices=True)

        candidate_base = flat * top_k
        token = tl.load(candidate_ptr + candidate_base + index)
        tl.store(tokens_ptr + flat, token)
        previous = index


def dflash2_greedy_sample(
    model,
    query_input_ids: torch.Tensor,
    hidden_states: torch.Tensor,
    num_speculative_tokens: int,
) -> torch.Tensor:
    """Return flattened DFlash2 draft tokens for the legacy proposer.

    The legacy DFlash query layout is [anchor, mask_0, ..., mask_N] per request.
    `hidden_states` contains one sample state per mask token, i.e. B * N rows.
    """
    if hidden_states.shape[0] % num_speculative_tokens != 0:
        raise ValueError(
            "DFlash2 sample hidden-state count must be divisible by "
            f"num_speculative_tokens={num_speculative_tokens}; "
            f"got {hidden_states.shape[0]}."
        )

    num_reqs = hidden_states.shape[0] // num_speculative_tokens
    selector = model.model.candidate_selector
    top_k = int(selector.top_k)

    candidate_ids, unary_logits = model.compute_candidates(hidden_states)
    candidate_ids = candidate_ids.view(num_reqs, num_speculative_tokens, top_k)
    unary_logits = unary_logits.view_as(candidate_ids)

    hidden = hidden_states.view(num_reqs, num_speculative_tokens, -1)
    query_stride = 1 + num_speculative_tokens
    anchor_indices = (
        torch.arange(num_reqs, dtype=torch.int64, device=query_input_ids.device)
        * query_stride
    )
    anchor_token_ids = query_input_ids[anchor_indices]

    scores = selector(
        candidate_ids,
        unary_logits,
        hidden,
        anchor_token_ids,
    ).contiguous()
    candidates = candidate_ids.contiguous()

    output = torch.empty(
        num_reqs * num_speculative_tokens,
        dtype=candidates.dtype,
        device=candidates.device,
    )
    block_k = triton.next_power_of_2(top_k)
    _selector_walk_greedy_kernel[(num_reqs,)](
        scores,
        candidates,
        output,
        num_steps=num_speculative_tokens,
        top_k=top_k,
        BLOCK_K=block_k,
        num_warps=1,
    )
    return output
