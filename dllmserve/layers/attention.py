import torch
from torch import nn
import triton
import triton.language as tl
import torch.nn.functional as F

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from dllmserve.utils.context import get_context


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    if slot_mapping.numel() != N:
        print(f"Warning: slot_mapping.numel {slot_mapping.numel()} != N {N}")
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D
    )


@triton.jit
def load_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)

    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + tl.arange(0, D)
    key = tl.load(k_cache_ptr + cache_offsets)
    value = tl.load(v_cache_ptr + cache_offsets)

    tl.store(key_ptr + key_offsets, key)
    tl.store(value_ptr + value_offsets, value)


def load_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    load_kvcache_kernel[(N,)](
        key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D
    )


class Attention(nn.Module):
    def __init__(self, num_heads, head_dim, scale, num_kv_heads, causal=True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.causal = causal
        self.k_cache = self.v_cache = torch.tensor([])

        self.layer_id = None

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)

        context = get_context()

        if context.is_diffusion:
            return self._mixed_sparse_dense_attention(q, k, v, context)
        else:
            if self.k_cache.numel() and self.v_cache.numel():
                store_kvcache(k, v, self.k_cache, self.v_cache, context.slot_mapping)
            return self._regular_attention(q, k, v, context)

    def _regular_attention(self, q, k, v, context):
        """Standard attention computation"""
        if context.is_prefill:
            if context.block_tables is not None:
                k, v = self.k_cache, self.v_cache
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=self.causal,
                block_table=context.block_tables,
            )
        else:
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),
                self.k_cache,
                self.v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=self.causal,
            )
        return o.view(-1, self.num_heads * self.head_dim)

    def _mixed_sparse_dense_attention(self, q, k, v, context):
        """
        Handle mixed batch with K/V caching instead of index caching.
        """

        seqs = context.seqs
        cu_seqlens_q = context.cu_seqlens_q
        cu_seqlens_k = context.cu_seqlens_k
        device = q.device
        B = len(seqs)

        Q_list_varlen = []
        K_list_varlen = []
        V_list_varlen = []
        output_positions_varlen = []

        slot_mapping = context.slot_mapping

        for i in range(B):
            seq = seqs[i]
            if seq.sparse_state is None or seq.sparse_state.is_full_attention:
                seq_start, seq_end = cu_seqlens_k[i], cu_seqlens_k[i + 1]
                seq_indices = torch.arange(seq_start, seq_end, device=device)

                Q_list_varlen.append(q[seq_indices])
                K_list_varlen.append(k[seq_indices])
                V_list_varlen.append(v[seq_indices])

                output_positions_varlen.append(seq_indices)
            elif seq.sparse_state.should_compute_cache:
                k_seq_start, k_seq_end = cu_seqlens_k[i], cu_seqlens_k[i + 1]
                seq_indices = torch.arange(k_seq_start, k_seq_end, device=device)

                Q_list_varlen.append(q[seq_indices])
                K_list_varlen.append(k[seq_indices])
                V_list_varlen.append(v[seq_indices])

                output_positions_varlen.append(seq_indices)

                # Cache import KV
                block_indices = (
                    seq.sparse_state.get_block_indices(
                        seq.num_prompt_tokens, seq.num_tokens
                    )
                    + k_seq_start
                )

                # Get prefix/suffix tokens
                all_indices = torch.arange(
                    k_seq_start, k_seq_end, dtype=torch.int64, device=device
                )
                prefix_suffix_mask = ~torch.isin(all_indices, block_indices)
                prefix_suffix_indices = all_indices[prefix_suffix_mask]

                # Compute importance and select K/V to cache
                q_block = q[block_indices]
                k_prefix_suffix = k[prefix_suffix_indices]
                v_prefix_suffix = v[prefix_suffix_indices]

                k_important = None
                v_important = None
                if seq.sparse_state.config.head_select:
                    # Select important tokens
                    k_important, v_important = self._select_important_head_kv(
                        q_block, k_prefix_suffix, v_prefix_suffix, seq
                    )
                else:
                    # Select important tokens
                    important_idx = self._select_important_kv(
                        q_block, k_prefix_suffix, seq
                    )

                    # Cache important K/V
                    k_important = k_prefix_suffix[important_idx]
                    v_important = v_prefix_suffix[important_idx]

                current_slot = slot_mapping[i]
                store_kvcache(
                    k_important,
                    v_important,
                    self.k_cache,
                    self.v_cache,
                    current_slot,
                )

            elif seq.sparse_state.should_reuse_cache:
                k_seq_start, k_seq_end = cu_seqlens_k[i], cu_seqlens_k[i + 1]
                block_indices = torch.arange(
                    k_seq_start, k_seq_end, dtype=torch.int64, device=device
                )

                cached_key = torch.zeros(
                    (seq.current_sparse_token, self.num_kv_heads, self.head_dim),
                    device=device,
                    dtype=k.dtype,
                )
                cached_value = torch.zeros(
                    (seq.current_sparse_token, self.num_kv_heads, self.head_dim),
                    device=device,
                    dtype=v.dtype,
                )

                current_slot = slot_mapping[i]
                load_kvcache(
                    cached_key,
                    cached_value,
                    self.k_cache,
                    self.v_cache,
                    current_slot,
                )

                Q_list_varlen.append(q[block_indices])
                K_list_varlen.append(torch.cat([cached_key, k[block_indices]], dim=0))
                V_list_varlen.append(torch.cat([cached_value, v[block_indices]], dim=0))

                output_positions_varlen.append(block_indices)

        Q_all_varlen = torch.cat(Q_list_varlen, dim=0).contiguous()
        V_all_varlen = torch.cat(V_list_varlen, dim=0).contiguous()
        K_all_varlen = torch.cat(K_list_varlen, dim=0).contiguous()

        q_lens_varlen = [_q.size(0) for _q in Q_list_varlen]
        k_lens_varlen = [_k.size(0) for _k in K_list_varlen]

        cu_q_varlen = torch.zeros(
            len(q_lens_varlen) + 1, dtype=torch.int32, device=device
        )
        cu_k_varlen = torch.zeros(
            len(k_lens_varlen) + 1, dtype=torch.int32, device=device
        )
        cu_q_varlen[1:] = torch.cumsum(
            torch.tensor(q_lens_varlen, device=device, dtype=torch.int32), dim=0
        )
        cu_k_varlen[1:] = torch.cumsum(
            torch.tensor(k_lens_varlen, device=device, dtype=torch.int32), dim=0
        )

        outputs = torch.zeros_like(q)

        attn_out = flash_attn_varlen_func(
            Q_all_varlen,
            K_all_varlen,
            V_all_varlen,
            cu_seqlens_q=cu_q_varlen,
            cu_seqlens_k=cu_k_varlen,
            max_seqlen_q=max(q_lens_varlen) if q_lens_varlen else 0,
            max_seqlen_k=max(k_lens_varlen) if k_lens_varlen else 0,
            softmax_scale=self.scale,
            causal=False,
        )

        offset = 0
        for seq_indices, q_len in zip(output_positions_varlen, q_lens_varlen):
            outputs[seq_indices] = attn_out[offset : offset + q_len]
            offset += q_len

        return outputs.view(-1, self.num_heads * self.head_dim)

    def _select_important_kv(self, q_block, k_candidates, seq):
        """
        Select important token indices based on attention scores.
        Returns indices (relative to k_candidates).
        """

        # Average query over tokens
        q_avg = q_block.mean(dim=0, keepdim=True)

        # Handle GQA
        k_exp = k_candidates
        if self.num_kv_heads < self.num_heads:
            repeat_factor = self.num_heads // self.num_kv_heads
            k_exp = k_candidates.repeat_interleave(repeat_factor, dim=1)

        # Compute importance scores
        scores = torch.einsum("qhd,khd->qk", q_avg, k_exp).squeeze(0) / (
            self.head_dim**0.5
        )

        # Apply max pooling if specified
        if seq.sparse_state.pool_kernel > 1 and scores.numel() > 0:
            scores = F.max_pool1d(
                scores.unsqueeze(0).unsqueeze(0),
                kernel_size=seq.sparse_state.pool_kernel,
                stride=1,
                padding=seq.sparse_state.pool_kernel // 2,
            ).squeeze(0).squeeze(0)[: scores.numel()]

        # Select top-k by ratio
        n_keep = max(1, int(k_candidates.size(0) * seq.sparse_state.retention_ratio))
        n_keep = min(n_keep, k_candidates.size(0))

        _, top_indices = torch.topk(scores, k=n_keep, dim=0)
        return top_indices

    def _select_important_head_kv(self, q_block, k_candidates, v_candidates, seq):
        """
        Select important K/V tokens per kv-head.

        Inputs:
        q_block:        [block_len, num_heads, head_dim]
        k_candidates:   [k_len,     num_kv_heads, head_dim]  (prefix+suffix only)
        v_candidates:   [k_len,     num_kv_heads, head_dim]

        Returns:
        k_sel:          [selected_k_len, num_kv_heads, head_dim]
        v_sel:          [selected_k_len, num_kv_heads, head_dim]
        """
        device = q_block.device
        Hq = self.num_heads
        Hkv = self.num_kv_heads
        D = self.head_dim
        Klen = k_candidates.size(0)

        if Klen == 0:
            return k_candidates, v_candidates

        # Average query over tokens within the current block -> [Hq, D]
        q_avg = q_block.mean(dim=0)

        # Rearrange K/V to [Hkv, Klen, D]
        k_perm = k_candidates.permute(1, 0, 2)
        v_perm = v_candidates.permute(1, 0, 2)

        # Compute per-kv-head scores over candidate tokens
        if Hq == Hkv:
            # scores[h, k] = q_avg[h] · k_perm[h, k]
            scores = torch.einsum("hd,hkd->hk", q_avg, k_perm) / (D**0.5)  # [Hkv, Klen]
        else:
            # GQA: group query heads that share one kv-head
            group = Hq // Hkv
            q_group = q_avg.view(Hkv, group, D)  # [Hkv, group, D]
            scores_g = torch.einsum("hgd,hkd->hgk", q_group, k_perm) / (
                D**0.5
            )  # [Hkv, group, Klen]
            scores = scores_g.max(dim=1).values  # [Hkv, Klen]

        # Optional smoothing/pooling per kv-head
        pool_k = seq.sparse_state.pool_kernel if (seq.sparse_state is not None) else 1
        if pool_k > 1 and Klen > 0:
            s = scores.unsqueeze(1)  # [Hkv, 1, Klen]
            s = F.max_pool1d(s, kernel_size=pool_k, stride=1, padding=pool_k // 2)
            scores = s.squeeze(1)  # [Hkv, Klen]
            if scores.dim() > 0:  # Ensure not a scalar
                scores = scores[..., :Klen]

        # Per-kv-head budget (same selected_k_len for all heads to keep a rectangular cache)
        ratio = (
            seq.sparse_state.retention_ratio if (seq.sparse_state is not None) else 0.5
        )
        selected_k_len = min(max(1, int(Klen * ratio)), Klen)

        # Top-k indices per kv-head
        top_idx = torch.topk(
            scores, k=selected_k_len, dim=1
        ).indices  # [Hkv, selected_k_len]
        idx_expanded = top_idx.unsqueeze(-1).expand(
            -1, -1, D
        )  # [Hkv, selected_k_len, D]

        # Gather per kv-head, then transpose back to [selected_k_len, Hkv, D]
        k_sel = torch.gather(k_perm, 1, idx_expanded).transpose(0, 1).contiguous()
        v_sel = torch.gather(v_perm, 1, idx_expanded).transpose(0, 1).contiguous()

        return k_sel, v_sel
