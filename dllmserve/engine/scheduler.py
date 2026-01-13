from collections import deque, defaultdict
import math

from dllmserve.config import Config
from dllmserve.engine.sequence import Sequence, SequenceStatus, ModelType
from dllmserve.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config, model_type: ModelType):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.model_type = model_type

        self.block_manager = BlockManager(
            config.num_kvcache_blocks, config.kvcache_block_size
        )

        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool | None]:
        return self._schedule_diffusion(), None

    def _schedule_diffusion(self) -> list[Sequence]:

        scheduled_seqs: list[Sequence] = []
        num_seqs = 0
        num_batched_tokens = 0

        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running[0]
            if (
                seq.sparse_state is None
                or seq.sparse_state.is_full_attention
                or seq.sparse_state.should_compute_cache
            ):
                num_tokens = len(seq)
            elif seq.sparse_state.should_reuse_cache:
                num_tokens = seq.sparse_state.block_len
            else:
                exit("unrecognized state")

            if num_batched_tokens + num_tokens > self.max_num_batched_tokens:
                # TODO: continue can add more reqs but violate fcfs, which is neccessary for our sys
                # TODO: implement skip operation
                break
                # continue
            num_seqs += 1
            num_batched_tokens += num_tokens
            # print("num_batched_tokens", num_batched_tokens)
            self.running.popleft()
            scheduled_seqs.append(seq)

        # === 2. If capacity left, move some waiting seqs into running ===
        remaining_slots = self.max_num_seqs - len(scheduled_seqs)
        while remaining_slots > 0 and self.waiting:
            seq = self.waiting[0]
            if seq.sparse_state is None:
                num_tokens = len(seq)
                pages_needed = 0
            elif (
                seq.sparse_state.is_full_attention
                or seq.sparse_state.should_compute_cache
            ):
                num_tokens = len(seq)
                pages_needed = seq.num_blocks
            elif seq.sparse_state.should_reuse_cache:
                num_tokens = seq.sparse_state.block_len
                pages_needed = seq.num_blocks
            else:
                exit("unrecognized state")

            if (
                num_batched_tokens + num_tokens > self.max_num_batched_tokens
                or not self.block_manager.can_allocate_sparse(seq)
            ):
                # TODO: continue can add more reqs but violate fcfs
                break

            self.block_manager.allocate_sparse(seq)
            remaining_slots -= 1
            num_batched_tokens += num_tokens

            self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            scheduled_seqs.append(seq)

        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs

    def preempt(self, seq: Sequence):
        # TODO: simple set to 0 may not good.
        seq.current_step = 0  # Reset progress

        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], result_data) -> None:
        if not seqs:
            return

        if result_data is None:  # Non-rank-0 processes
            return

        updated_token_sequences = result_data
        for seq, updated_tokens in zip(seqs, updated_token_sequences):
            seq.token_ids = updated_tokens
            seq.current_step += 1
            if seq.sparse_state is not None:
                seq.sparse_state.advance_step()

            if seq.current_step >= seq.sampling_params.steps:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
