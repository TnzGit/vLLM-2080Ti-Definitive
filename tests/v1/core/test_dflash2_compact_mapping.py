import torch

from vllm.v1.core.dflash2_flex_attention_compat import (
    _compact_single_request_kv,
    _remap_block_indices_into,
    prepare_dflash2_compact_mapping,
)
from vllm.v1.spec_decode.dflash import _resolve_dflash_draft_cudagraph


class _BlockMask:
    def __init__(self):
        self.seq_lengths = torch.tensor([8], dtype=torch.int32)
        self.kv_indices = torch.tensor([[2, 0]], dtype=torch.int32)
        self.full_kv_indices = torch.tensor([[2, 0]], dtype=torch.int32)


class _Metadata:
    num_reqs = 1
    block_size = 4
    max_seq_len = 8
    max_possible_sequence_length = 8
    total_cache_tokens = 16
    num_blocks = 2

    def __init__(self):
        self.block_mask = _BlockMask()
        self.block_table = torch.tensor([[2, 0]], dtype=torch.int32)
        self.physical_to_logical = torch.tensor([[-1, 7, 0, -1]], dtype=torch.int64)


def test_mapping_prepared_once_and_reused_by_layers():
    metadata = _Metadata()
    prepare_dflash2_compact_mapping(metadata)
    assert metadata._dflash2_compact_state["prepare_count"] == 1

    kv = torch.arange(48, dtype=torch.float16).reshape(2, 4, 2, 3, 1)
    kv = kv.transpose(2, 3)
    compact_ptrs = []
    for _ in range(5):
        compact, restore = _compact_single_request_kv(kv, metadata)
        assert compact is not None
        compact_ptrs.append(compact.untyped_storage().data_ptr())
        restore()

    assert len(set(compact_ptrs)) == 1
    assert metadata._dflash2_compact_state["prepare_count"] == 1


def test_mapping_refreshes_when_new_proposal_metadata_changes():
    metadata = _Metadata()
    prepare_dflash2_compact_mapping(metadata)
    metadata.block_table[0, :2] = torch.tensor([3, 1])
    metadata.block_mask = _BlockMask()
    prepare_dflash2_compact_mapping(metadata)
    state = metadata._dflash2_compact_state
    assert state["prepare_count"] == 1
    assert state["block_ids"][:2].tolist() == [3, 1]


def test_remap_into_reuses_output_and_preserves_invalid_entries():
    indices = torch.tensor([[-1, 2, 0]], dtype=torch.int32)
    remap = torch.tensor([5, 6, 7], dtype=torch.int64)
    output = torch.empty_like(indices)

    result = _remap_block_indices_into(indices, remap, output)

    assert result is output
    assert result.tolist() == [[-1, 7, 5]]


def test_dflash_draft_cudagraph_auto_disables_sm75(monkeypatch):
    monkeypatch.delenv("VLLM_DFLASH_DRAFT_CUDAGRAPH", raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (7, 5))

    enabled, reason = _resolve_dflash_draft_cudagraph(torch.device("cuda"))

    assert enabled is False
    assert reason == "auto-sm75"


def test_dflash_draft_cudagraph_can_be_forced_on(monkeypatch):
    monkeypatch.setenv("VLLM_DFLASH_DRAFT_CUDAGRAPH", "1")
    enabled, reason = _resolve_dflash_draft_cudagraph(torch.device("cuda"))

    assert enabled is True
    assert reason == "forced-on"
