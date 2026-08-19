# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.model_executor.models import ModelRegistry
from vllm.model_executor.models.qwen3_dflash2 import _grouped_conv, _score_edges
from vllm.triton_utils import triton
from vllm.v1.spec_decode.dflash2 import _selector_walk_greedy_kernel


@pytest.mark.parametrize("block_size", [5, 8])
def test_grouped_conv_matches_reference(block_size: int):
    torch.manual_seed(0)
    batch, taps, num_groups, group_size = 3, 3, 4, 2
    hidden = torch.randn(batch * block_size, num_groups * group_size)
    delta = torch.randn(batch * block_size, taps, num_groups)
    base = torch.randn(taps, num_groups * group_size)

    actual = _grouped_conv(
        hidden, delta, base, block_size, num_groups, group_size, taps
    )
    hidden_blocks = hidden.view(batch, block_size, num_groups, group_size)
    expected = torch.zeros_like(hidden_blocks)
    base = base.view(taps, num_groups, group_size)
    delta = delta.view(batch, block_size, taps, num_groups)
    for position in range(block_size):
        for tap in range(min(taps, position + 1)):
            expected[:, position] += (
                base[tap] + delta[:, position, tap, :, None]
            ) * hidden_blocks[:, position - tap]

    torch.testing.assert_close(actual, expected.flatten(0, 1).flatten(-2))


def test_selector_edges_match_sequential_reference():
    torch.manual_seed(1)
    batch, steps, top_k, rank = 2, 4, 3, 5
    vocab = 17
    predecessors = torch.randn(vocab, rank)
    successors = torch.randn(vocab, rank)
    candidate_ids = torch.randint(vocab, (batch, steps, top_k))
    unary = torch.randn(batch, steps, top_k)
    hidden = torch.randn(batch, steps, rank)
    anchors = torch.randint(vocab, (batch,))

    actual = _score_edges(
        predecessors,
        successors,
        candidate_ids,
        unary,
        hidden,
        anchors,
        top_k,
    )
    expected = torch.empty_like(actual)
    for step in range(steps):
        pred = (
            anchors[:, None].expand(-1, top_k)
            if step == 0
            else candidate_ids[:, step - 1]
        )
        expected[:, step] = unary[:, step, None] + torch.einsum(
            "bpr,bcr->bpc",
            predecessors[pred] * hidden[:, step, None],
            successors[candidate_ids[:, step]],
        )

    torch.testing.assert_close(actual, expected)


def test_dflash2_architecture_is_registered():
    assert "DFlash2DraftModel" in ModelRegistry.get_supported_archs()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/Triton")
def test_selector_walk_is_request_local_and_sanitizes_nan():
    # [request, step, predecessor_candidate, successor_candidate]
    scores = torch.tensor(
        [
            [
                [[float("nan"), 2.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [5.0, float("nan"), 7.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[float("inf"), 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[-3.0, -2.0, -1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
        ],
        dtype=torch.float32,
        device="cuda",
    )
    candidates = torch.tensor(
        [
            [[10, 11, 12], [20, 21, 22]],
            [[30, 31, 32], [40, 41, 42]],
        ],
        dtype=torch.int64,
        device="cuda",
    )
    output = torch.empty(4, dtype=torch.int64, device="cuda")

    _selector_walk_greedy_kernel[(2,)](
        scores,
        candidates,
        output,
        num_steps=2,
        top_k=3,
        BLOCK_K=triton.next_power_of_2(3),
        num_warps=1,
    )

    # Request 0: NaN is ignored -> candidate 11, then predecessor row 1 -> 22.
    # Request 1: +inf wins -> candidate 30, then predecessor row 0 -> 42.
    assert output.view(2, 2).cpu().tolist() == [[11, 22], [30, 42]]
