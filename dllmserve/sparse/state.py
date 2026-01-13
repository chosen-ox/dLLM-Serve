from dataclasses import dataclass
from typing import Optional, Tuple, List
import torch


@dataclass
class SparseConfig:
    """Global configuration for Sparse-dLLM"""

    enabled: bool = False
    default_block_len: int = 32
    retention_ratio: float = 0.5
    pool_kernel: int = 3
    delay_step: int = 1
    head_select: bool = False


@dataclass
class SparseContext:
    """Context for sparse attention with per-request state"""

    cache_state: int
    block_indices: torch.Tensor  # Indices for current block (never cached)
    block_spans: torch.Tensor  # [1, 2] (start, length) for this sequence
    retention_ratio: float
    pool_kernel: int
    seq_sparse_state: Optional["PerRequestSparseState"] = (
        None  # Reference to sequence state
    )


class PerRequestSparseState:
    """Per-request sparse state managing K/V cache"""

    def __init__(self, seq_id: int, config: SparseConfig, device: torch.device):
        self.seq_id = seq_id
        self.config = config

        self.current_block_idx = 0
        self.step_in_block = 0
        self.cache_state = 0
        self.device = device

    def compute_block_boundaries(
        self, prompt_len: int, total_len: int
    ) -> tuple[int, int]:
        """
        Compute start and length of current block.
        Returns: (block_start, block_length)
        """
        gen_len = total_len - prompt_len
        if gen_len <= 0:
            return (prompt_len, 0)

        # Start of current block (relative to prompt end)
        block_start = prompt_len + self.current_block_idx * self.block_len

        # Length of current block (handle last block being shorter)
        remaining = total_len - block_start
        block_length = min(self.block_len, remaining)

        return (block_start, block_length)

    def get_cache_state(self) -> int:
        """
        Determine cache state for current step.
        0: Full attention (no caching)
        1: Compute and cache indices
        2: Reuse cached indices
        """

        if not self.config.enabled:
            return 0

        return self.cache_state

    def advance_step(self):
        """Advance to next step within block"""
        self.step_in_block += 1

        # For last block, this is not right
        if self.step_in_block % self.block_len == 0:
            self.step_in_block = 0
            self.current_block_idx += 1

        # update cache state
        if self.config.delay_step == 1:
            if self.step_in_block == 0:
                # if self.step_in_block == 0 and self.current_block_idx == 0:
                self.cache_state = 0
            elif self.step_in_block == 1:
                # elif self.step_in_block == 1 and self.current_block_idx == 0:
                self.cache_state = 1
            else:
                self.cache_state = 2
        elif self.config.delay_step == 0:
            self.cache_state = 1 if self.step_in_block == 0 else 2
        else:
            if self.step_in_block < self.config.delay_step:
                self.cache_state = 0
            elif self.step_in_block == self.config.delay_step:
                self.cache_state = 1
            else:
                self.cache_state = 2

    # return current block indices in current seq
    def get_block_indices(self, prompt_len, total_len):
        block_start, block_len = self.compute_block_boundaries(prompt_len, total_len)
        self.block_indices = torch.arange(
            block_start,
            block_start + block_len,
            dtype=torch.int64,
            device=self.device,
        )
        return self.block_indices

    def get_unuse_block_indices(self, prompt_len, total_len):
        block_start, block_len = self.compute_block_boundaries(prompt_len, total_len)
        self.block_indices_prefix = torch.arange(
            0,
            block_start,
            dtype=torch.int64,
            device=self.device,
        )
        self.block_indices_suffix = torch.arange(
            block_start + block_len,
            total_len,
            dtype=torch.int64,
            device=self.device,
        )

        self.block_indices = torch.cat(
            [self.block_indices_prefix, self.block_indices_suffix], dim=0
        )

        return self.block_indices

    @property
    def pool_kernel(self):
        return self.config.pool_kernel

    @property
    def retention_ratio(self):
        return self.config.retention_ratio

    @property
    def block_len(self):
        return self.config.default_block_len

    @property
    def should_compute_cache(self) -> bool:
        """Should compute cache for prefix/suffix (not current block)"""
        return self.cache_state == 1

    @property
    def should_reuse_cache(self) -> bool:
        """Should reuse cached indices for prefix/suffix"""
        return self.cache_state == 2

    @property
    def is_full_attention(self) -> bool:
        """Should use full attention without any caching"""
        return self.cache_state == 0
