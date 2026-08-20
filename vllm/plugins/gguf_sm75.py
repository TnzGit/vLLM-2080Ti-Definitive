# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GGUF compatibility layer for the SM75/vLLM 0.21 fork.

The vLLM 0.21 base still contains the in-tree GGUF loader and CUDA kernels.
This module keeps those implementations and adds only the model-family glue
needed by Qwen3.5-compatible dense models (including Qwen3.6/Qwen3.8):

* Qwen3.5 GGUF tensor-name mapping for the GatedDeltaNet (GDN) backbone.
* GGML grouped-value-head layout restoration without dequantizing packed
  weights.
* TP-aware sharding for the packed GDN output projection.
* Rebuilding vocab embedding / LM-head modules when the model constructor did
  not pass the GGUF quantization config.

The first-stage scope is intentionally text-only dense Qwen3.5-compatible
models. MoE and multimodal GGUF loading remain on the original in-tree path
until they receive their own validation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

import gguf
import torch
from torch import nn
from torch.nn.parameter import UninitializedParameter

from vllm.config import ModelConfig, VllmConfig
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
from vllm.model_executor.model_loader.gguf_loader import (
    GGUFModelLoader as InTreeGGUFModelLoader,
)
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.model_executor.model_loader.weight_utils import (
    get_gguf_weight_type_map,
    gguf_quant_weights_iterator,
    gguf_quant_weights_iterator_multi,
)
from vllm.model_executor.models.utils import WeightsMapper, maybe_prefix
from vllm.model_executor.utils import set_weight_attrs
from vllm.utils.torch_utils import set_default_torch_dtype

if TYPE_CHECKING:
    from transformers import PretrainedConfig

logger = init_logger(__name__)

_QWEN35_DENSE_MODEL_TYPES = {"qwen3_5", "qwen3_5_text"}
_QWEN35_MOE_MODEL_TYPES = {"qwen3_5_moe", "qwen3_5_moe_text"}

_QWEN35_ATTN_SUBSTR: dict[str, str] = {
    "attn_norm.": "input_layernorm.",
    "post_attention_norm.": "post_attention_layernorm.",
    "attn_q_norm.": "self_attn.q_norm.",
    "attn_k_norm.": "self_attn.k_norm.",
    "attn_q.": "self_attn.q_proj.",
    "attn_k.": "self_attn.k_proj.",
    "attn_v.": "self_attn.v_proj.",
    "attn_output.": "self_attn.o_proj.",
    "attn_qkv.": "linear_attn.in_proj_qkv.",
    "attn_gate.": "linear_attn.in_proj_z.",
    "ssm_alpha.": "linear_attn.in_proj_a.",
    "ssm_beta.": "linear_attn.in_proj_b.",
    "ssm_conv1d.": "linear_attn.conv1d.",
    "ssm_norm.": "linear_attn.norm.",
    "ssm_out.": "linear_attn.out_proj.",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_a.weight": "linear_attn.A_log",
    "ssm_a": "linear_attn.A_log",
}

_QWEN35_DENSE_SUBSTR: dict[str, str] = {
    "ffn_gate.": "mlp.gate_proj.",
    "ffn_up.": "mlp.up_proj.",
    "ffn_down.": "mlp.down_proj.",
}


@dataclass(frozen=True, slots=True)
class GGUFHeadTilingLayout:
    """Grouped-head layout used by GGML for Qwen3.5 GDN value heads."""

    heads_per_group: int
    head_dim: int

    def __post_init__(self) -> None:
        if self.heads_per_group <= 1:
            raise ValueError("heads_per_group must be greater than one")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")

    def input_to_gguf(self, x: torch.Tensor) -> torch.Tensor:
        """Reorder a local vLLM activation shard into GGML head-tile order."""
        num_heads, remainder = divmod(x.shape[-1], self.head_dim)
        if remainder or num_heads % self.heads_per_group:
            raise ValueError(
                "Cannot reorder linear input shape "
                f"{tuple(x.shape)} with heads_per_group={self.heads_per_group}, "
                f"head_dim={self.head_dim}"
            )
        num_groups = num_heads // self.heads_per_group
        shape = (*x.shape[:-1], num_groups, self.heads_per_group, self.head_dim)
        return x.reshape(shape).transpose(-3, -2).reshape(x.shape).contiguous()

    def weight_to_vllm(
        self,
        weight: torch.Tensor,
        *,
        dim: int,
        head_dim: int | None = None,
    ) -> torch.Tensor:
        """Restore a non-packed GGML head-tiled dimension to vLLM order."""
        if dim < 0:
            dim += weight.ndim
        if not 0 <= dim < weight.ndim:
            raise ValueError(f"Invalid dimension {dim} for shape {tuple(weight.shape)}")

        head_dim = self.head_dim if head_dim is None else head_dim
        if head_dim <= 0:
            raise ValueError("head_dim must be positive")
        num_heads, remainder = divmod(weight.shape[dim], head_dim)
        num_groups, group_remainder = divmod(num_heads, self.heads_per_group)
        if remainder or group_remainder:
            raise ValueError(
                "Cannot restore GGML weight shape "
                f"{tuple(weight.shape)} along dim={dim} with "
                f"heads_per_group={self.heads_per_group}, head_dim={head_dim}"
            )

        shape = list(weight.shape)
        tiled_shape = (
            *shape[:dim],
            self.heads_per_group,
            num_groups,
            head_dim,
            *shape[dim + 1 :],
        )
        return (
            weight.reshape(tiled_shape)
            .transpose(dim, dim + 1)
            .reshape(shape)
            .contiguous()
        )

    def shard_weight(
        self,
        weight: torch.Tensor,
        *,
        dim: int,
        logical_size: int,
        block_size: int,
        tp_rank: int,
        tp_size: int,
    ) -> torch.Tensor:
        """Select this TP rank's groups while keeping GGUF blocks packed."""
        if tp_size == 1:
            return weight

        num_groups, remainder = divmod(
            logical_size, self.head_dim * self.heads_per_group
        )
        if remainder:
            raise ValueError(
                f"Cannot shard logical input size {logical_size} with "
                f"heads_per_group={self.heads_per_group}, "
                f"head_dim={self.head_dim}"
            )
        local_groups, remainder = divmod(num_groups, tp_size)
        if remainder:
            raise ValueError(
                f"Cannot divide {num_groups} head groups across TP size {tp_size}"
            )
        if (local_groups * self.head_dim) % block_size:
            raise ValueError(
                f"TP size {tp_size} splits a stored head tile at "
                f"{local_groups * self.head_dim} logical elements, which is "
                f"not aligned to GGML block size {block_size}"
            )

        total_groups = self.heads_per_group * num_groups
        group_span, remainder = divmod(weight.shape[dim], total_groups)
        if remainder:
            raise ValueError(
                f"Packed weight dimension {weight.shape[dim]} is not "
                f"divisible into {total_groups} head groups"
            )

        moved = weight.movedim(dim, 0)
        tiled = moved.reshape(
            self.heads_per_group, num_groups, group_span, *moved.shape[1:]
        )
        shard = tiled[:, tp_rank * local_groups : (tp_rank + 1) * local_groups]
        return shard.reshape(-1, *moved.shape[1:]).movedim(0, dim).contiguous()


def _layout_aware_row_weight_loader(
    layer: nn.Module,
    layout: GGUFHeadTilingLayout,
    logical_input_size: int,
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
) -> None:
    """Shard a packed GDN out-projection without breaking GGML head tiles."""
    qweight_type = int(layer.qweight_type.weight_type)
    if qweight_type not in gguf.GGML_QUANT_SIZES:
        raise ValueError(f"Unknown GGUF quantization type id {qweight_type}")
    block_size, _ = gguf.GGML_QUANT_SIZES[qweight_type]

    tp_rank = int(getattr(layer, "tp_rank", 0))
    tp_size = int(getattr(layer, "tp_size", 1))
    loaded_weight = layout.shard_weight(
        loaded_weight,
        dim=1,
        logical_size=logical_input_size,
        block_size=block_size,
        tp_rank=tp_rank,
        tp_size=tp_size,
    )

    if isinstance(param, UninitializedParameter):
        param.materialize(loaded_weight.shape, dtype=loaded_weight.dtype)
    if param.size() != loaded_weight.size():
        raise ValueError(
            "GGUF layout-aware shard shape mismatch: "
            f"parameter={tuple(param.shape)} loaded={tuple(loaded_weight.shape)}"
        )
    param.data.copy_(loaded_weight)


class LayoutAwareGGUFLinearMethod(InTreeGGUFLinearMethod):
    """In-tree GGUF linear method with an optional GGML activation layout."""

    def __init__(
        self,
        quant_config: "SM75GGUFConfig",
        layout: GGUFHeadTilingLayout,
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
        # Qwen3.5 currently registers a layout only for GDN out_proj, which is
        # a row-parallel linear. The wrapper shards the packed input dimension
        # by GGML head groups instead of taking one contiguous TP slice.
        set_weight_attrs(
            layer.qweight,
            {
                "weight_loader": partial(
                    _layout_aware_row_weight_loader,
                    layer,
                    self.layout,
                    input_size,
                ),
                "gguf_layout": self.layout,
                "gguf_logical_input_size": input_size,
            },
        )

    def apply(
        self,
        layer: nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return super().apply(layer, self.layout.input_to_gguf(x), bias)


class SM75GGUFConfig(InTreeGGUFConfig):
    """GGUF config carrying per-linear GGML layouts into model construction."""

    def __init__(self, unquantized_modules: list[str] | None = None) -> None:
        super().__init__(unquantized_modules)
        self.linear_layouts: dict[str, GGUFHeadTilingLayout] = {}

    def get_quant_method(
        self, layer: nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        method = super().get_quant_method(layer, prefix)
        layout = self.linear_layouts.get(prefix)
        if layout is not None and isinstance(layer, LinearBase):
            if not isinstance(method, InTreeGGUFLinearMethod):
                raise TypeError(
                    f"GGUF layout registered for non-GGUF linear {prefix}: "
                    f"{type(method).__name__}"
                )
            return LayoutAwareGGUFLinearMethod(self, layout)
        return method

    def register_linear_layouts(
        self,
        layouts: Mapping[str, GGUFHeadTilingLayout],
        prefix: str = "",
    ) -> None:
        self.linear_layouts.update(
            (maybe_prefix(prefix, module_name), layout)
            for module_name, layout in layouts.items()
        )

    def apply_vllm_mapper(self, hf_to_vllm_mapper: WeightsMapper) -> None:
        super().apply_vllm_mapper(hf_to_vllm_mapper)
        if self.linear_layouts:
            layouts = self.linear_layouts
            mapped_names = hf_to_vllm_mapper.apply_list(list(layouts))
            self.linear_layouts = dict(
                zip(mapped_names, layouts.values(), strict=True)
            )


def _recursive_replace_vocab_modules(
    model: nn.Module,
    quant_config: SM75GGUFConfig,
    prefix: str = "",
) -> None:
    """Rebuild vocab modules whose Qwen constructor omitted quant_config."""
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
                params_dtype=child_module.params_dtype,
                org_num_embeddings=child_module.org_vocab_size,
                padding_size=child_module.padding_size,
                quant_config=quant_config,
                prefix=qual_name,
                **kwargs,
            )
            replacements[id(child_module)] = replacement
            setattr(module, child_name, replacement)

    replace(model, prefix)


def _qwen35_layer_substr() -> dict[str, str]:
    return _QWEN35_ATTN_SUBSTR | _QWEN35_DENSE_SUBSTR


def build_qwen35_text_mapper(*, nested_multimodal_config: bool) -> WeightsMapper:
    """Build the GGUF->HF name mapper for the dense Qwen3.5 text backbone."""
    backbone_prefix = (
        "model.language_model." if nested_multimodal_config else "model."
    )
    return WeightsMapper(
        orig_to_new_prefix={
            "token_embd.": backbone_prefix + "embed_tokens.",
            "blk.": backbone_prefix + "layers.",
            "output_norm.": backbone_prefix + "norm.",
            "output.": "lm_head.",
        },
        orig_to_new_substr=_qwen35_layer_substr(),
    )


def _map_tensor_name(mapper: WeightsMapper, name: str) -> str | None:
    mapped = mapper.apply_list([name])[0]
    return mapped if mapped != name else None


def _gdn_value_head_layout(
    text_config: "PretrainedConfig",
) -> GGUFHeadTilingLayout | None:
    num_key_heads = getattr(text_config, "linear_num_key_heads", 0) or 0
    num_value_heads = getattr(text_config, "linear_num_value_heads", 0) or 0
    if not num_key_heads or not num_value_heads:
        return None
    repeat, remainder = divmod(num_value_heads, num_key_heads)
    if remainder or repeat <= 1:
        return None
    return GGUFHeadTilingLayout(
        heads_per_group=repeat,
        head_dim=int(text_config.linear_value_head_dim),
    )


def _tensor_names(gguf_files: Iterable[str]) -> list[str]:
    names: list[str] = []
    for gguf_file in gguf_files:
        names.extend(tensor.name for tensor in gguf.GGUFReader(gguf_file).tensors)
    return names


def build_qwen35_name_map(
    tensor_names: Iterable[str],
    *,
    nested_multimodal_config: bool,
) -> dict[str, str]:
    """Map text-backbone GGUF tensor names to Qwen3.5 HF weight names."""
    mapper = build_qwen35_text_mapper(
        nested_multimodal_config=nested_multimodal_config
    )
    name_map: dict[str, str] = {}
    unmapped: list[str] = []
    for name in sorted(tensor_names):
        # First-stage scope: text backbone only. Vision/projector and MTP
        # sidecars are deliberately left for later phases.
        if name.startswith(("v.", "mm.")) or ".nextn." in name:
            continue
        mapped = _map_tensor_name(mapper, name)
        if mapped is None:
            unmapped.append(name)
        else:
            name_map[name] = mapped
    if unmapped:
        logger.warning(
            "No Qwen3.5 HF name for %d GGUF tensor(s), skipping: %s",
            len(unmapped),
            unmapped,
        )
    return name_map


def _transform_qwen35_weight(
    name: str,
    weight: torch.Tensor,
    text_config: "PretrainedConfig",
    layout: GGUFHeadTilingLayout | None,
) -> torch.Tensor:
    """Undo llama.cpp Qwen3.5 GDN layout transforms for one mapped tensor."""
    if name.endswith("qweight_type"):
        return weight

    base = name.removesuffix(".qweight").removesuffix(".weight")
    if layout is not None:
        num_key_heads = int(text_config.linear_num_key_heads)
        key_dim = int(text_config.linear_key_head_dim)
        if base.endswith("linear_attn.out_proj") and name.endswith(".weight"):
            return layout.weight_to_vllm(weight, dim=1)
        if base.endswith(".A_log"):
            return layout.weight_to_vllm(torch.log(-weight), dim=0, head_dim=1)
        if base.endswith(".dt_bias"):
            return layout.weight_to_vllm(weight, dim=0, head_dim=1)
        if base.endswith("linear_attn.in_proj_z"):
            return layout.weight_to_vllm(weight, dim=0)
        if base.endswith(("linear_attn.in_proj_a", "linear_attn.in_proj_b")):
            return layout.weight_to_vllm(weight, dim=0, head_dim=1)
        if base.endswith("linear_attn.in_proj_qkv"):
            qk_rows = key_dim * num_key_heads * 2
            value = layout.weight_to_vllm(weight[qk_rows:], dim=0)
            return torch.cat([weight[:qk_rows], value], dim=0)
        if base.endswith("linear_attn.conv1d") and weight.dim() == 2:
            qk_rows = key_dim * num_key_heads * 2
            value = layout.weight_to_vllm(weight[qk_rows:], dim=0)
            return torch.cat([weight[:qk_rows], value], dim=0).unsqueeze(1)

    if base.endswith(".A_log"):
        return torch.log(-weight)
    if (
        name.endswith("norm.weight")
        and not name.endswith("linear_attn.norm.weight")
    ):
        return weight - 1
    if "conv1d.weight" in name and weight.dim() == 2:
        return weight.unsqueeze(1)
    if name.endswith(".weight") and weight.dim() == 1 and "norm" not in name:
        return weight.unsqueeze(0)
    return weight


def _transform_qwen35_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    text_config: "PretrainedConfig",
    layout: GGUFHeadTilingLayout | None,
) -> Iterable[tuple[str, torch.Tensor]]:
    for name, weight in weights:
        yield name, _transform_qwen35_weight(name, weight, text_config, layout)


def _is_qwen35_dense(model_config: ModelConfig) -> bool:
    return model_config.hf_text_config.model_type in _QWEN35_DENSE_MODEL_TYPES


def _is_qwen35_moe(model_config: ModelConfig) -> bool:
    return model_config.hf_text_config.model_type in _QWEN35_MOE_MODEL_TYPES


def _has_nested_multimodal_config(model_config: ModelConfig) -> bool:
    return (
        model_config.hf_config is not model_config.hf_text_config
        and getattr(model_config.hf_config, "vision_config", None) is not None
    )


def _require_first_stage_scope(model_config: ModelConfig) -> None:
    if _is_qwen35_moe(model_config):
        raise NotImplementedError(
            "The first SM75 GGUF stage supports dense Qwen3.5-compatible models "
            "only; MoE GGUF support is not enabled yet."
        )
    if not _is_qwen35_dense(model_config):
        return
    if _has_nested_multimodal_config(model_config):
        mm_config = model_config.multimodal_config
        if mm_config is None or not mm_config.language_model_only:
            raise NotImplementedError(
                "The first SM75 Qwen3.5/3.6 GGUF stage is text-only. "
                "Start the model with --language-model-only; multimodal mmproj "
                "loading will be added in a later stage."
            )


class SM75GGUFModelLoader(InTreeGGUFModelLoader):
    """In-tree GGUF loader with dense Qwen3.5-compatible text adaptation."""

    def _prepare_qwen35(
        self,
        model_config: ModelConfig,
    ) -> tuple[
        str,
        list[str],
        dict[str, str],
        GGUFHeadTilingLayout | None,
        dict[str, GGUFHeadTilingLayout],
    ]:
        local_model_path = self._prepare_weights(model_config)
        gguf_files = self._get_all_gguf_files(local_model_path)
        nested = _has_nested_multimodal_config(model_config)
        tensor_names = _tensor_names(gguf_files)
        name_map = build_qwen35_name_map(
            tensor_names,
            nested_multimodal_config=nested,
        )
        if not name_map:
            raise RuntimeError("No Qwen3.5-compatible GGUF text weights were mapped")

        # Qwen3.5-family GGUFs omit output.weight when embeddings are tied.
        model_config.hf_text_config.update(
            {"tie_word_embeddings": "output.weight" not in tensor_names}
        )

        layout = _gdn_value_head_layout(model_config.hf_text_config)
        linear_layouts: dict[str, GGUFHeadTilingLayout] = {}
        if layout is not None:
            linear_layouts = {
                mapped_name.removesuffix(".weight"): layout
                for mapped_name in name_map.values()
                if mapped_name.endswith("linear_attn.out_proj.weight")
            }
        return local_model_path, gguf_files, name_map, layout, linear_layouts

    @staticmethod
    def _qwen35_weight_type_map(
        gguf_files: Iterable[str],
        name_map: dict[str, str],
    ) -> dict[str, str]:
        weight_type_map: dict[str, str] = {}
        for gguf_file in gguf_files:
            weight_type_map.update(get_gguf_weight_type_map(gguf_file, name_map))
        return weight_type_map

    @staticmethod
    def _qwen35_weights_iterator(
        gguf_files: list[str],
        name_map: dict[str, str],
        text_config: "PretrainedConfig",
        layout: GGUFHeadTilingLayout | None,
    ) -> Iterable[tuple[str, torch.Tensor]]:
        if len(gguf_files) > 1:
            weights = gguf_quant_weights_iterator_multi(gguf_files, name_map)
        else:
            weights = gguf_quant_weights_iterator(gguf_files[0], name_map)
        return _transform_qwen35_weights(weights, text_config, layout)

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        _require_first_stage_scope(model_config)
        if not _is_qwen35_dense(model_config):
            return super().load_weights(model, model_config)
        _, gguf_files, name_map, layout, _ = self._prepare_qwen35(model_config)
        model.load_weights(
            self._qwen35_weights_iterator(
                gguf_files,
                name_map,
                model_config.hf_text_config,
                layout,
            )
        )

    def load_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        prefix: str = "",
    ) -> nn.Module:
        _require_first_stage_scope(model_config)
        if not _is_qwen35_dense(model_config):
            return super().load_model(vllm_config, model_config, prefix)

        _, gguf_files, name_map, layout, linear_layouts = self._prepare_qwen35(
            model_config
        )
        weight_type_map = self._qwen35_weight_type_map(gguf_files, name_map)
        unquantized_modules = [
            name.removesuffix(".weight")
            for name, weight_type in weight_type_map.items()
            if weight_type in ("F32", "F16", "BF16") and name.endswith(".weight")
        ]

        quant_config = cast(SM75GGUFConfig, vllm_config.quant_config)
        if not isinstance(quant_config, SM75GGUFConfig):
            raise RuntimeError(
                "SM75 GGUF plugin was loaded after the GGUF quant config was "
                "constructed; restart vLLM so the plugin can register first."
            )
        quant_config.unquantized_modules.extend(unquantized_modules)
        quant_config.register_linear_layouts(linear_layouts, prefix=prefix)

        target_device = torch.device(vllm_config.device_config.device)
        with set_default_torch_dtype(model_config.dtype):
            with target_device:
                model = initialize_model(
                    vllm_config=vllm_config,
                    model_config=model_config,
                    prefix=prefix,
                )
                _recursive_replace_vocab_modules(model, quant_config, prefix=prefix)

            model.load_weights(
                self._qwen35_weights_iterator(
                    gguf_files,
                    name_map,
                    model_config.hf_text_config,
                    layout,
                )
            )
            process_weights_after_loading(model, model_config, target_device)
        return model


def register() -> None:
    """Register the SM75 GGUF compatibility layer."""
    register_quantization_config("gguf")(SM75GGUFConfig)
    register_model_loader("gguf")(SM75GGUFModelLoader)
    logger.info_once(
        "Registered SM75 GGUF compatibility layer for dense Qwen3.5-family models."
    )
