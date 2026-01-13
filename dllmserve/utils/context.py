from dataclasses import dataclass
import torch
from typing import Optional


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    extra_slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    is_diffusion: bool = False
    seqs: Optional[list] = None


_CONTEXT = Context()


def get_context():
    return _CONTEXT


def set_context(
    is_prefill,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    max_seqlen_q=0,
    max_seqlen_k=0,
    slot_mapping=None,
    extra_slot_mapping=None,
    context_lens=None,
    block_tables=None,
    is_diffusion=False,
    seqs=None,
):
    """
    Backward-compatible context setter.

    - Keeps the original dataclass fields unchanged.
    - Any extra keyword args are attached as dynamic attributes on the Context object.
      This lets us pass sparse-dLLM metadata (e.g., block_indices) later without touching the dataclass.
    - Existing callers (both positional and keyword) remain unaffected.
    """
    global _CONTEXT
    ctx = Context(
        is_prefill=is_prefill,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        slot_mapping=slot_mapping,
        extra_slot_mapping=extra_slot_mapping,
        context_lens=context_lens,
        block_tables=block_tables,
        is_diffusion=is_diffusion,
        seqs=seqs,
    )
    _CONTEXT = ctx


def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
