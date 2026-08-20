# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM 0.21 compatibility fixes for the first SM75 Qwen GGUF stage.

This module is the registered plugin entrypoint for the first-stage GGUF
backport. It layers two vLLM-0.21-specific fixes on top of ``gguf_sm75``:

* Qwen3.5 GDN ``in_proj_qkv`` is loaded into the fused ``in_proj_qkvz``
  projection with shard id ``(0, 1, 2)``. The old in-tree GGUF loader rejects
  tuple shard ids, so split the fused on-disk tensor into its logical output
  shards and load each shard through the existing, TP-aware loader.
* vLLM 0.21's vocab modules do not retain ``params_dtype`` as an attribute.
  Infer it from the existing parameter when rebuilding embeddings/LM heads for
  packed GGUF loading.
"""

from __future__ import annotations

from functools import partial

import torch
from torch import nn

from vllm.logger import init_logger
from vllm.model_executor.layers.linear import LinearBase
from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizeMethodBase
from vllm.model_executor.layers.quantization.gguf import (
    GGUFConfig as InTreeGGUFConfig,
)
from vllm.model_executor.layers.quantization.gguf import (
    GGUFLinearMethod as InTreeGGUFLinearMethod,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.utils import set_weight_attrs

from . import gguf_sm75 as base

logger = init_logger(__name__)

ShardId = int | str


def _call_weight_loader(
    loader,
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    shard_id: ShardId | None = None,
) -> None:
    if shard_id is None:
        loader(param, loaded_weight)
    else:
        loader(param, loaded_weight, shard_id)


def _tuple_and_layout_aware_weight_loader(
    layer: nn.Module,
    fallback_loader,
    layout: base.GGUFHeadTilingLayout | None,
    logical_input_size: int,
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    loaded_shard_id: tuple[int, ...] | ShardId | None = None,
) -> None:
    """Bridge tuple-shard Qwen GDN weights to the vLLM 0.21 GGUF loader."""
    is_gguf_weight = getattr(param, "is_gguf_weight", False)
    is_gguf_weight_type = getattr(param, "is_gguf_weight_type", False)

    if isinstance(loaded_shard_id, tuple) and (
        is_gguf_weight or is_gguf_weight_type
    ):
        # Weight-type metadata is one scalar for the whole fused on-disk
        # tensor. Every logical shard receives the same type id.
        if is_gguf_weight_type:
            for shard_id in loaded_shard_id:
                _call_weight_loader(
                    fallback_loader,
                    param,
                    loaded_weight,
                    shard_id,
                )
            return

        output_dim = getattr(param, "output_dim", None)
        if output_dim is None:
            raise ValueError("Tuple-sharded GGUF weight is missing output_dim")
        if not hasattr(layer, "output_sizes"):
            raise TypeError(
                "Tuple-sharded GGUF loading requires a merged-column layer"
            )

        offset = 0
        for shard_id in loaded_shard_id:
            shard_size = int(layer.output_sizes[shard_id])
            loaded_shard = loaded_weight.narrow(output_dim, offset, shard_size)
            _call_weight_loader(
                fallback_loader,
                param,
                loaded_shard,
                shard_id,
            )
            offset += shard_size
        if offset != loaded_weight.shape[output_dim]:
            raise ValueError(
                "Fused GGUF tensor size does not match tuple shard sizes: "
                f"loaded={loaded_weight.shape[output_dim]} expected={offset} "
                f"shards={loaded_shard_id}"
            )
        return

    if layout is not None and is_gguf_weight and loaded_shard_id is None:
        base._layout_aware_row_weight_loader(
            layer,
            layout,
            logical_input_size,
            param,
            loaded_weight,
        )
        return

    if isinstance(loaded_shard_id, tuple):
        raise TypeError(
            f"Unexpected tuple shard id for non-GGUF parameter: {loaded_shard_id}"
        )
    _call_weight_loader(
        fallback_loader,
        param,
        loaded_weight,
        loaded_shard_id,
    )


class Stage1GGUFLinearMethod(InTreeGGUFLinearMethod):
    """In-tree GGUF linear method with Qwen3.5 tuple/layout compatibility."""

    def __init__(
        self,
        quant_config: "Stage1GGUFConfig",
        layout: base.GGUFHeadTilingLayout | None = None,
    ) -> None:
        super().__init__(quant_config)
        self.layout = layout

    def create_weights(
        self,
        layer: nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        super().create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )
        fallback_loader = extra_weight_attrs.get("weight_loader")
        if fallback_loader is None:
            fallback_loader = getattr(layer, "weight_loader", None)
        if fallback_loader is None:
            raise RuntimeError(
                f"No base weight loader available for GGUF layer {layer}"
            )

        wrapper = partial(
            _tuple_and_layout_aware_weight_loader,
            layer,
            fallback_loader,
            self.layout,
            input_size,
        )
        set_weight_attrs(layer.qweight, {"weight_loader": wrapper})
        set_weight_attrs(layer.qweight_type, {"weight_loader": wrapper})

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.layout is not None:
            x = self.layout.input_to_gguf(x)
        return super().apply(layer, x, bias)


class Stage1GGUFConfig(base.SM75GGUFConfig):
    """GGUF config with tuple-aware vLLM 0.21 linear loading."""

    def get_quant_method(
        self, layer: nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        # Bypass base.SM75GGUFConfig.get_quant_method() here so all quantized
        # linear layers get the tuple-aware method, not only GDN out_proj.
        method = InTreeGGUFConfig.get_quant_method(self, layer, prefix)
        if isinstance(layer, LinearBase) and isinstance(
            method, InTreeGGUFLinearMethod
        ):
            return Stage1GGUFLinearMethod(
                self,
                layout=self.linear_layouts.get(prefix),
            )
        return method


def _vocab_params_dtype(module: nn.Module) -> torch.dtype:
    params_dtype = getattr(module, "params_dtype", None)
    if isinstance(params_dtype, torch.dtype):
        return params_dtype
    parameter = next(module.parameters(recurse=False), None)
    if parameter is not None:
        return parameter.dtype
    return torch.get_default_dtype()


def _recursive_replace_vocab_modules(
    model: nn.Module,
    quant_config: Stage1GGUFConfig,
    prefix: str = "",
) -> None:
    """Rebuild vocab modules with GGUF quantization on vLLM 0.21."""
    replacements: dict[int, nn.Module] = {}

    def replace(module: nn.Module, module_prefix: str) -> None:
        for child_name, child_module in tuple(module._modules.items()):
            if child_module is None:
                continue
            qual_name = maybe_prefix(module_prefix, child_name)
            replacement = replacements.get(id(child_module))
            if replacement is not None:
                setattr(module, child_name, replacement)
                continue

            if type(child_module) not in (VocabParallelEmbedding, ParallelLMHead):
                replace(child_module, qual_name)
                continue

            expected_method = quant_config.get_quant_method(child_module, qual_name)
            if type(child_module.quant_method) is type(expected_method):
                continue

            kwargs = (
                {"bias": child_module.bias is not None}
                if type(child_module) is ParallelLMHead
                else {}
            )
            replacement = type(child_module)(
                child_module.num_embeddings,
                child_module.embedding_dim,
                params_dtype=_vocab_params_dtype(child_module),
                org_num_embeddings=child_module.org_vocab_size,
                padding_size=child_module.padding_size,
                quant_config=quant_config,
                prefix=qual_name,
                **kwargs,
            )
            replacements[id(child_module)] = replacement
            setattr(module, child_name, replacement)

    replace(model, prefix)


def register() -> None:
    """Register the complete first-stage SM75 GGUF compatibility layer."""
    # The loader in gguf_sm75 resolves these module globals when load_model()
    # executes, so replace them before registering the loader/config classes.
    base.SM75GGUFConfig = Stage1GGUFConfig
    base._recursive_replace_vocab_modules = _recursive_replace_vocab_modules

    register_quantization_config("gguf")(Stage1GGUFConfig)
    register_model_loader("gguf")(base.SM75GGUFModelLoader)
    logger.info_once(
        "Registered first-stage SM75 GGUF support for dense Qwen3.5-family models."
    )
