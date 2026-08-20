# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DFlash2 activation-dtype compatibility for quantized target hidden states."""

from __future__ import annotations

from typing import Any

import torch


def _match_linear_activation_dtype(
    hidden_states: torch.Tensor,
    linear: Any,
) -> torch.Tensor:
    """Cast activations to the dtype a vLLM LinearBase was constructed for.

    ``LinearBase.params_dtype`` is the activation/parameter compute dtype even
    when a quantization method stores weights in another representation. Using
    it instead of ``linear.weight.dtype`` avoids incorrectly casting activations
    to an integer/packed weight dtype for quantized linear implementations.
    """
    params_dtype = getattr(linear, "params_dtype", None)
    if params_dtype is None or hidden_states.dtype == params_dtype:
        return hidden_states
    if not hidden_states.is_floating_point():
        raise TypeError(
            "DFlash2 combine_hidden_states expected floating-point target hidden "
            f"states, got {hidden_states.dtype}."
        )
    return hidden_states.to(dtype=params_dtype)


def combine_dflash2_hidden_states(
    model: Any,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Combine target aux states using the draft FC's compute dtype.

    TurboQuant target paths in this fork can expose FP32 auxiliary hidden
    states while the native DFlash2 draft FC is constructed in FP16/BF16.
    PyTorch GEMM requires matching activation and weight compute dtypes, so
    normalize the boundary before entering the draft FC. The FC output then
    naturally stays in the draft model dtype for context-KV precomputation and
    the subsequent draft forward.
    """
    if not model.model.use_aux_hidden_state:
        return hidden_states

    needs_squeeze = hidden_states.dim() == 1
    if needs_squeeze:
        hidden_states = hidden_states.unsqueeze(0)

    fc = model.model.fc
    hidden_states = _match_linear_activation_dtype(hidden_states, fc)
    result = fc(hidden_states)

    if needs_squeeze:
        result = result.squeeze(0)
    return result
