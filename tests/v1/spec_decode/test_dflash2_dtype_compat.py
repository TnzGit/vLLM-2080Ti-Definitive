# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.models.dflash2_dtype_compat import (
    _match_linear_activation_dtype,
    combine_dflash2_hidden_states,
)


@pytest.mark.parametrize("draft_dtype", [torch.float16, torch.bfloat16])
def test_match_linear_activation_dtype_uses_params_dtype(draft_dtype) -> None:
    # Model the real failure boundary: target aux states arrive as FP32 while
    # the native draft FC was constructed in FP16/BF16.
    hidden = torch.randn(3, 8, dtype=torch.float32)
    fake_linear = SimpleNamespace(
        params_dtype=draft_dtype,
        # Deliberately make a fake packed-weight dtype different from params_dtype
        # to ensure the compatibility helper does not key off weight dtype.
        weight=torch.empty(1, dtype=torch.int8),
    )

    cast = _match_linear_activation_dtype(hidden, fake_linear)

    assert cast.dtype == draft_dtype
    assert cast.shape == hidden.shape


def test_match_linear_activation_dtype_is_noop_when_already_matching() -> None:
    hidden = torch.randn(2, 4, dtype=torch.float16)
    fake_linear = SimpleNamespace(params_dtype=torch.float16)

    cast = _match_linear_activation_dtype(hidden, fake_linear)

    assert cast is hidden


def test_combine_hidden_states_casts_before_fc_and_preserves_1d_shape() -> None:
    events: list[torch.dtype] = []

    class FakeFC:
        params_dtype = torch.float16

        def __call__(self, hidden_states: torch.Tensor) -> torch.Tensor:
            events.append(hidden_states.dtype)
            assert hidden_states.dtype == torch.float16
            return hidden_states + 1

    draft = SimpleNamespace(
        model=SimpleNamespace(
            use_aux_hidden_state=True,
            fc=FakeFC(),
        )
    )
    hidden = torch.randn(8, dtype=torch.float32)

    result = combine_dflash2_hidden_states(draft, hidden)

    assert events == [torch.float16]
    assert result.shape == hidden.shape
    assert result.dtype == torch.float16


def test_combine_hidden_states_without_aux_is_unchanged() -> None:
    draft = SimpleNamespace(model=SimpleNamespace(use_aux_hidden_state=False))
    hidden = torch.randn(2, 8, dtype=torch.float32)

    result = combine_dflash2_hidden_states(draft, hidden)

    assert result is hidden
