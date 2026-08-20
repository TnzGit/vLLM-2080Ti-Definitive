# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""vLLM 0.21 compatibility fixes for the first SM75 Qwen GGUF stage.

This module is the registered plugin entrypoint for the first-stage GGUF
backport. It layers vLLM-0.21-specific compatibility on top of ``gguf_sm75``:

* Qwen3.5 GDN ``in_proj_qkv`` is loaded into the fused ``in_proj_qkvz``
  projection with shard id ``(0, 1, 2)``. The old in-tree GGUF loader rejects
  tuple shard ids, so split the fused on-disk tensor into its logical output
  shards and load each shard through the existing, TP-aware loader.
* Detach TP-local staged GGUF shards from the full CPU tensor storage retained
  by ``Tensor.narrow()`` views. This keeps ``data_container`` host memory local
  to each TP rank instead of pinning the whole pre-shard tensor in every worker.
* vLLM 0.21's vocab modules do not retain ``params_dtype`` as an attribute.
  Infer it from the existing parameter when rebuilding embeddings/LM heads for
  packed GGUF loading.
* Keep the Hugging Face config source separate from the GGUF weight source.
  This lets ``--hf-config-path Qwen/Qwen3.6-27B`` drive model/processor config
  while ``model_weights`` remains ``repo:quant`` or a local GGUF path.
"""

from __future__ import annotations

import os
from functools import partial, wraps

import torch
from huggingface_hub import hf_hub_download
from torch import nn

from vllm.engine.arg_utils import EngineArgs
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
from vllm.model_executor.model_loader.weight_utils import download_gguf
from vllm.model_executor.models.utils import maybe_prefix
from vllm.model_executor.utils import set_weight_attrs
from vllm.transformers_utils.gguf_utils import is_gguf

from . import gguf_sm75 as base

logger = init_logger(__name__)

ShardId = int | str
_MIB = 1024**2
_GIB = 1024**3
_LOG_HOST_STAGING = os.getenv("VLLM_GGUF_LOG_HOST_STAGING", "0") == "1"
_HOST_STAGING_AVOIDED_BYTES = 0
_HOST_STAGING_LOCAL_BYTES = 0
_HOST_STAGING_NEXT_LOG_BYTES = 512 * _MIB


def _is_gguf_reference(value: str | None) -> bool:
    if not value:
        return False
    # vLLM 0.21's is_gguf() recognizes local files and repo:quant, while the
    # in-tree loader also accepts exact remote repo/path/file.gguf references.
    return value.endswith(".gguf") or is_gguf(value)


def _storage_nbytes(tensor: torch.Tensor) -> int:
    """Return bytes owned by the tensor's underlying storage."""
    return int(tensor.untyped_storage().nbytes())


def _detach_staged_gguf_views(param: torch.Tensor) -> tuple[int, int]:
    """Compact CPU GGUF staging views into independent TP-local storages.

    vLLM 0.21 stores fused GGUF shards in ``param.data_container``. The stock
    TP loaders use ``narrow()`` to select a rank-local slice and append that
    view. A narrow view still owns a reference to the full pre-shard CPU
    storage, so one small staged shard can keep a much larger ``torch.tensor``
    allocation alive until ``process_weights_after_loading()``.

    Group views by backing storage, clone each group member into compact CPU
    storage, and report the amount of backing storage that no longer needs to
    remain reachable plus the TP-local bytes deliberately kept for later fused
    parameter materialization.
    """
    if not getattr(param, "is_gguf_weight", False):
        return 0, 0

    data_container = getattr(param, "data_container", None)
    if not isinstance(data_container, list) or not data_container:
        return 0, 0

    # Only compact views whose storage is larger than the tensor itself. Fully
    # owned compact tensors are already safe and copying them would only raise
    # peak RSS without releasing anything.
    groups: dict[tuple[int, int], list[int]] = {}
    for index, tensor in enumerate(data_container):
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
            continue
        tensor_bytes = tensor.numel() * tensor.element_size()
        storage_bytes = _storage_nbytes(tensor)
        if storage_bytes <= tensor_bytes:
            continue
        storage_ptr = int(tensor.untyped_storage().data_ptr())
        groups.setdefault((storage_ptr, storage_bytes), []).append(index)

    avoided_bytes = 0
    local_bytes = 0
    for (_, storage_bytes), indices in groups.items():
        group_local_bytes = 0
        replacements: list[tuple[int, torch.Tensor]] = []
        for index in indices:
            staged = data_container[index]
            group_local_bytes += staged.numel() * staged.element_size()
            replacements.append(
                (
                    index,
                    staged.clone(memory_format=torch.contiguous_format),
                )
            )

        # Replace only after all views in the group have been cloned so the
        # shared backing storage stays valid throughout the copy.
        for index, detached in replacements:
            data_container[index] = detached

        avoided_bytes += max(storage_bytes - group_local_bytes, 0)
        local_bytes += group_local_bytes

    return avoided_bytes, local_bytes


def _record_host_staging_detach(avoided_bytes: int, local_bytes: int) -> None:
    """Optionally emit coarse per-worker evidence for the host-memory fix."""
    global _HOST_STAGING_AVOIDED_BYTES
    global _HOST_STAGING_LOCAL_BYTES
    global _HOST_STAGING_NEXT_LOG_BYTES

    if avoided_bytes <= 0:
        return

    _HOST_STAGING_AVOIDED_BYTES += avoided_bytes
    _HOST_STAGING_LOCAL_BYTES += local_bytes
    if not _LOG_HOST_STAGING:
        return

    if _HOST_STAGING_AVOIDED_BYTES >= _HOST_STAGING_NEXT_LOG_BYTES:
        logger.info(
            "GGUF host staging detached %.2f GiB of over-retained CPU backing "
            "storage so far; %.2f GiB of TP-local staged shards are retained.",
            _HOST_STAGING_AVOIDED_BYTES / _GIB,
            _HOST_STAGING_LOCAL_BYTES / _GIB,
        )
        while _HOST_STAGING_NEXT_LOG_BYTES <= _HOST_STAGING_AVOIDED_BYTES:
            _HOST_STAGING_NEXT_LOG_BYTES += 512 * _MIB


def _detach_and_record_staging(param: torch.Tensor) -> None:
    avoided_bytes, local_bytes = _detach_staged_gguf_views(param)
    _record_host_staging_detach(avoided_bytes, local_bytes)


def _call_weight_loader(
    loader,
    param: torch.Tensor,
    loaded_weight: torch.Tensor,
    shard_id: ShardId | None = None,
    *,
    detach_staging: bool = True,
) -> None:
    if shard_id is None:
        loader(param, loaded_weight)
    else:
        loader(param, loaded_weight, shard_id)

    # The in-tree loader may have just appended one or more TP narrow views to
    # data_container. For ordinary loads, detach immediately. Tuple GDN loads
    # defer this until all logical shards have been appended so a shared backing
    # storage is accounted for and cloned only as one group.
    if detach_staging:
        _detach_and_record_staging(param)


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
                detach_staging=False,
            )
            offset += shard_size
        if offset != loaded_weight.shape[output_dim]:
            raise ValueError(
                "Fused GGUF tensor size does not match tuple shard sizes: "
                f"loaded={loaded_weight.shape[output_dim]} expected={offset} "
                f"shards={loaded_shard_id}"
            )
        _detach_and_record_staging(param)
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


def _gguf_config_source(
    model: str,
    tokenizer: str | None,
    hf_config_path: str | None,
) -> str | None:
    del model
    if hf_config_path:
        return hf_config_path
    if tokenizer and not _is_gguf_reference(tokenizer):
        return tokenizer
    return None


def _patch_engine_args() -> None:
    """Route model config to HF while preserving the GGUF ref as weights."""
    if getattr(EngineArgs, "_sm75_gguf_model_config_patched", False):
        return

    original_create_model_config = EngineArgs.create_model_config

    @wraps(original_create_model_config)
    def create_model_config(self, *args, **kwargs):
        gguf_model = self.model
        if _is_gguf_reference(gguf_model):
            config_source = _gguf_config_source(
                gguf_model,
                self.tokenizer if isinstance(self.tokenizer, str) else None,
                self.hf_config_path,
            )
            if config_source is not None:
                self.quantization = "gguf"
                self.load_format = "gguf"
                self.model_weights = gguf_model
                if self.served_model_name is None:
                    self.served_model_name = gguf_model
                self.model = config_source
                if self.tokenizer is None:
                    self.tokenizer = config_source
        return original_create_model_config(self, *args, **kwargs)

    EngineArgs.create_model_config = create_model_config
    EngineArgs._sm75_gguf_model_config_patched = True


class Stage1GGUFModelLoader(base.SM75GGUFModelLoader):
    """GGUF loader that reads weights from ModelConfig.model_weights."""

    def _prepare_weights(self, model_config):
        model_ref = getattr(model_config, "model_weights", None)
        if not isinstance(model_ref, str) or not _is_gguf_reference(model_ref):
            return super()._prepare_weights(model_config)

        if os.path.isfile(model_ref):
            return model_ref
        if "/" in model_ref and model_ref.endswith(".gguf"):
            repo_id, filename = model_ref.rsplit("/", 1)
            return hf_hub_download(repo_id=repo_id, filename=filename)
        if "/" in model_ref and ":" in model_ref:
            repo_id, quant_type = model_ref.rsplit(":", 1)
            return download_gguf(
                repo_id,
                quant_type,
                cache_dir=self.load_config.download_dir,
                revision=model_config.revision,
                ignore_patterns=self.load_config.ignore_patterns,
            )
        raise ValueError(
            f"Unrecognised GGUF weight reference: {model_ref} "
            "(expected local file, <repo_id>/<filename>.gguf, "
            "or <repo_id>:<quant_type>)"
        )


def register() -> None:
    """Register the complete first-stage SM75 GGUF compatibility layer."""
    # The loader in gguf_sm75 resolves these module globals when load_model()
    # executes, so replace them before registering the loader/config classes.
    base.SM75GGUFConfig = Stage1GGUFConfig
    base._recursive_replace_vocab_modules = _recursive_replace_vocab_modules
    _patch_engine_args()

    register_quantization_config("gguf")(Stage1GGUFConfig)
    register_model_loader("gguf")(Stage1GGUFModelLoader)
    logger.info_once(
        "Registered first-stage SM75 GGUF support for dense Qwen3.5-family models."
    )
