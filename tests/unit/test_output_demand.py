"""Output-demand indexing and cache preservation, using a CPU attention oracle."""

from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

import dllmserve.layers.attention as attention_module
from dllmserve.models.dream import DreamForMaskedLM
from dllmserve.sparse.state import PerRequestSparseState, SparseConfig
from dllmserve.utils.context import reset_context, set_context
from dllmserve.utils.output_demand import final_layer_query_ranges


def sequence(prompt, total, phase=None, block=32, block_idx=0):
    state = None
    if phase is not None:
        state = PerRequestSparseState(
            0, SparseConfig(enabled=True, default_block_len=block), torch.device("cpu")
        )
        state.cache_state = phase
        state.current_block_idx = block_idx
    return SimpleNamespace(
        num_prompt_tokens=prompt, num_tokens=total, sparse_state=state, current_sparse_token=64
    )


@pytest.mark.parametrize(
    "prompt,block,block_idx,total",
    [
        (0, 32, 0, 96),
        (129, 32, 0, 193),
        (256, 32, 1, 352),
        (257, 64, 1, 337),
    ],
)
def test_shifted_block_rows_are_kept(prompt, block, block_idx, total):
    seq = sequence(prompt, total, 1, block, block_idx)
    ranges = final_layer_query_ranges([seq])
    begin, length = seq.sparse_state.compute_block_boundaries(prompt, total)
    kept = set(range(total)) if ranges is None else set(range(*ranges[0]))
    assert {max(0, i - 1) for i in range(begin, begin + length)} <= kept


def test_mixed_packed_offsets_and_reuse():
    seqs = [sequence(256, 320, 1), sequence(100, 148, 2, 16), sequence(513, 545)]
    assert final_layer_query_ranges(seqs) == [(128, 288), (320, 336), (720, 881)]
    assert final_layer_query_ranges([sequence(100, 132, 2)]) is None
    assert final_layer_query_ranges([]) is None


def test_empty_output_falls_back():
    assert final_layer_query_ranges([sequence(256, 256)]) is None


def reference_attention(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale, **kwargs):
    output = []
    group = q.shape[1] // k.shape[1]
    for i in range(len(cu_seqlens_q) - 1):
        qq = q[cu_seqlens_q[i] : cu_seqlens_q[i + 1]].transpose(0, 1)
        kk = k[cu_seqlens_k[i] : cu_seqlens_k[i + 1]].repeat_interleave(group, 1).transpose(0, 1)
        vv = v[cu_seqlens_k[i] : cu_seqlens_k[i + 1]].repeat_interleave(group, 1).transpose(0, 1)
        output.append(
            (torch.softmax(qq @ kk.transpose(-1, -2) * softmax_scale, -1) @ vv).transpose(0, 1)
        )
    return torch.cat(output)


@pytest.fixture
def cpu_attention(monkeypatch):
    monkeypatch.setattr(attention_module, "flash_attn_varlen_func", reference_attention)
    monkeypatch.setattr(
        attention_module,
        "store_kvcache",
        lambda k, v, ck, cv, slots: (
            ck.index_copy_(0, slots.long(), k),
            cv.index_copy_(0, slots.long(), v),
        ),
    )
    monkeypatch.setattr(
        attention_module,
        "load_kvcache",
        lambda k, v, ck, cv, slots: (k.copy_(ck[slots.long()]), v.copy_(cv[slots.long()])),
    )
    # These tests isolate row demand from compiler/backend numerical differences.
    monkeypatch.setattr(torch._dynamo.config, "disable", True)
    yield
    reset_context()


@pytest.mark.parametrize("head_select", [False, True])
def test_compact_attention_keeps_cache_selection(cpu_attention, head_select):
    torch.manual_seed(7)
    seqs = [sequence(256, 320, 1), sequence(128, 144, 2, 16)]
    seqs[0].sparse_state.config.head_select = head_select
    q = torch.randn(336, 4, 8, dtype=torch.float64)
    k = torch.randn(336, 2, 8, dtype=torch.float64)
    v = torch.randn_like(k)
    attn = attention_module.Attention(4, 8, 8**-0.5, 2, causal=False)
    attn.k_cache = torch.randn(256, 2, 8, dtype=q.dtype)
    attn.v_cache = torch.randn_like(attn.k_cache)
    # 288 non-block candidates, with retention ratio 0.5.
    slots = [torch.arange(144), torch.arange(192, 256)]
    set_context(
        True,
        cu_seqlens_k=torch.tensor([0, 320, 336]),
        is_diffusion=True,
        seqs=seqs,
        slot_mapping=slots,
    )
    full = attn(q, k, v)
    ck, cv = attn.k_cache.clone(), attn.v_cache.clone()
    ranges = final_layer_query_ranges(seqs)
    indices = torch.cat([torch.arange(a, b) for a, b in ranges])
    compact = attn(q, k, v, query_ranges=ranges)
    torch.testing.assert_close(compact, full[indices], rtol=1e-12, atol=1e-12)
    assert torch.equal(attn.k_cache, ck) and torch.equal(attn.v_cache, cv)


def test_last_layer_preserves_shifted_consumers(cpu_attention):
    torch.manual_seed(9)
    config = PretrainedConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        max_position_embeddings=1024,
        rope_theta=10000.0,
        vocab_size=32,
    )
    model = DreamForMaskedLM(config).double().eval()
    for name, param in model.named_parameters():
        if name.endswith("weight") and param.ndim == 1:
            param.data.fill_(1)
        else:
            param.data.normal_(0, 0.1)
    seqs = [sequence(256, 320, 0), sequence(129, 177)]
    tokens = torch.randint(0, 32, (497,))
    positions = torch.cat([torch.arange(320), torch.arange(177)])
    common = dict(
        cu_seqlens_k=torch.tensor([0, 320, 497]),
        is_diffusion=True,
        seqs=seqs,
        slot_mapping=[[], []],
    )
    with torch.inference_mode():
        set_context(True, **common)
        full = model(tokens, positions)
        ranges = final_layer_query_ranges(seqs)
        indices = torch.cat([torch.arange(a, b) for a, b in ranges])
        set_context(
            True, **common, final_layer_query_ranges=ranges, final_layer_output_indices=indices
        )
        compact = model(tokens, positions)
    required = torch.tensor(list(range(255, 287)) + list(range(448, 496)))
    torch.testing.assert_close(compact[required], full[required], rtol=2e-6, atol=2e-6)
    assert compact.shape == full.shape
