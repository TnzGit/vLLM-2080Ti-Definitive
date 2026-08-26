# SPDX-License-Identifier: Apache-2.0

from math import prod

import torch

from vllm.third_party.flash_linear_attention.ops import chunk as chunk_module
from vllm.third_party.flash_linear_attention.ops.chunk import (
    _gdn_prefill_workspace_specs,
    get_gdn_post_conv_workspace,
)
from vllm.v1.worker.workspace import (
    init_workspace_manager,
    reset_workspace_manager,
)


def test_gdn_prefill_workspace_covers_full_scratch_lifetime() -> None:
    specs = _gdn_prefill_workspace_specs(
        batch_size=1,
        num_tokens=2048,
        num_heads=4,
        key_dim=128,
        value_dim=128,
        dtype=torch.float16,
        num_chunks=32,
        num_sequences=1,
    )

    assert specs == (
        ((1, 2048, 4), torch.float32),
        ((1, 2048, 4, 64), torch.float32),
        ((1, 2048, 4, 64), torch.float16),
        ((1, 2048, 4, 128), torch.float16),
        ((1, 2048, 4, 128), torch.float16),
        ((1, 32, 4, 128, 128), torch.float16),
        ((1, 2048, 4, 128), torch.float16),
        ((1, 4, 128, 128), torch.float32),
    )


def test_gdn_prefill_workspace_scales_linearly_with_token_budget() -> None:
    def size(num_tokens: int, num_chunks: int) -> int:
        return sum(
            prod(shape) * dtype.itemsize
            for shape, dtype in _gdn_prefill_workspace_specs(
                batch_size=1,
                num_tokens=num_tokens,
                num_heads=4,
                key_dim=128,
                value_dim=128,
                dtype=torch.float16,
                num_chunks=num_chunks,
                num_sequences=1,
            )
        )

    fixed_final_state_bytes = 4 * 128 * 128 * 4
    size_2048 = size(2048, 32)
    size_4096 = size(4096, 64)
    assert size_4096 - fixed_final_state_bytes == 2 * (
        size_2048 - fixed_final_state_bytes
    )


def test_gdn_prefill_pipeline_wires_every_scratch_output(monkeypatch) -> None:
    reset_workspace_manager()
    init_workspace_manager(torch.device("cpu"))
    seen: dict[str, torch.Tensor] = {}

    def cumsum(g, **kwargs):
        seen["g"] = kwargs["out"]
        return kwargs["out"]

    def dot(**kwargs):
        seen["A"] = kwargs["out"]
        return kwargs["out"]

    def solve(**kwargs):
        seen["Ai"] = kwargs["out"]
        return kwargs["out"]

    def wy(**kwargs):
        seen["w"] = kwargs["w_out"]
        seen["u"] = kwargs["u_out"]
        return kwargs["w_out"], kwargs["u_out"]

    def delta(**kwargs):
        seen["h"] = kwargs["h_out"]
        seen["v_new"] = kwargs["v_new_out"]
        seen["final_state"] = kwargs["final_state_out"]
        return (
            kwargs["h_out"],
            kwargs["v_new_out"],
            kwargs["final_state_out"],
        )

    monkeypatch.setattr(chunk_module, "chunk_local_cumsum", cumsum)
    monkeypatch.setattr(chunk_module, "chunk_scaled_dot_kkt_fwd", dot)
    monkeypatch.setattr(chunk_module, "solve_tril", solve)
    monkeypatch.setattr(chunk_module, "recompute_w_u_fwd", wy)
    monkeypatch.setattr(chunk_module, "chunk_gated_delta_rule_fwd_h", delta)
    monkeypatch.setattr(
        chunk_module,
        "chunk_fwd_o",
        lambda **kwargs: kwargs["core_attn_out"],
    )

    prep = get_gdn_post_conv_workspace(
        num_tokens=64,
        num_key_heads=4,
        num_value_heads=4,
        key_dim=128,
        value_dim=128,
        dtype=torch.float16,
    )
    assert prep is not None
    prep_ptrs = {tensor.data_ptr() for tensor in prep}
    assert len(prep_ptrs) == len(prep)
    q, k, v, g, beta = (tensor.unsqueeze(0) for tensor in prep)
    state = torch.empty((1, 4, 128, 128), dtype=torch.float32)
    output = torch.empty_like(v)
    _, actual_output, _, final_state, *_ = chunk_module.chunk_gated_delta_rule_fwd(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=1.0,
        initial_state=state,
        output_final_state=True,
        core_attn_out=output,
    )

    assert actual_output is output
    assert final_state is seen["final_state"]
    assert set(seen) == {
        "g",
        "A",
        "Ai",
        "w",
        "u",
        "h",
        "v_new",
        "final_state",
    }
    assert all(tensor is not None for tensor in seen.values())
    assert prep_ptrs.isdisjoint(tensor.data_ptr() for tensor in seen.values())
    reset_workspace_manager()
