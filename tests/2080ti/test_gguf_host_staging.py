# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.plugins.gguf_sm75_stage1 import (
    _call_weight_loader,
    _storage_nbytes,
)


def test_standard_tp_weight_loader_detaches_appended_view() -> None:
    full = torch.arange(16 * 4, dtype=torch.uint8).reshape(16, 4)
    full_storage_ptr = full.untyped_storage().data_ptr()
    param = SimpleNamespace(
        is_gguf_weight=True,
        data_container=[],
        shard_id=[],
        shard_id_map={},
    )

    def fallback(target, loaded_weight, shard_id=None):
        # Model the ordinary vLLM 0.21 Merged/QKV TP path: select one rank via
        # narrow() and retain that view for later fused parameter materialization.
        local = loaded_weight.narrow(0, 0, loaded_weight.shape[0] // 2)
        target.shard_id_map[shard_id] = len(target.data_container)
        target.shard_id.append(shard_id)
        target.data_container.append(local)

    _call_weight_loader(fallback, param, full, 0)

    assert param.shard_id == [0]
    assert len(param.data_container) == 1
    staged = param.data_container[0]
    assert tuple(staged.shape) == (8, 4)
    assert staged.untyped_storage().data_ptr() != full_storage_ptr
    assert _storage_nbytes(staged) == staged.numel() * staged.element_size()
    torch.testing.assert_close(staged, full[:8])
