import torch

from vllm.v1.core.dflash2_flex_attention_compat import (
    _compact_single_request_kv,
    prepare_dflash2_compact_mapping,
)


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
