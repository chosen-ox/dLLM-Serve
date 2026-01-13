from copy import copy
from enum import Enum, auto
from itertools import count
import math
from dllmserve.sparse.state import PerRequestSparseState


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


# identity model type at runtime
class ModelType(Enum):
    DIFFUSION = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(
        self,
        token_ids: list[int],
        sampling_params,
        config,
        sparse_configs,
    ):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.config = config
        self.model_type = self.config.model_type
        self.torch_device = self.config.torch_device

        # Diffusion model initialization
        self.num_prompt_tokens = len(token_ids)
        # Get mask_token_id from config instead of hardcoded value
        mask_id = self.config.hf_config.mask_token_id
        gen_length = sampling_params.gen_length
        self.token_ids = copy(token_ids) + [mask_id] * gen_length
        self.num_tokens = len(self.token_ids)
        self.sampling_params = sampling_params
        self.current_step: int = 0

        if sparse_configs is not None:
            self.sparse_state = PerRequestSparseState(
                self.seq_id, sparse_configs, self.torch_device
            )
        else:
            self.sparse_state = None

        # for dllm, if do not use sparse, num_cached_tokens is num_tokens
        self.num_cached_tokens = 0
        self.block_table = []
        self.pre_num_blocks = 0

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def prompt_token_ids(self):
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens :]

    @property
    def num_completion_tokens(self):
        return self.sampling_params.gen_length

    @property
    def num_cached_blocks(self):
        if self.sparse_state.should_reuse_cache:
            return self.current_sparse_token // self.block_size
        raise NotImplementedError

    @property
    def num_should_cached_blocks(self):
        if (
            self.sparse_state.should_compute_cache
            or self.sparse_state.should_reuse_cache
        ):
            return (
                self.current_sparse_token + self.block_size - 1
            ) // self.block_size

        raise NotImplementedError

    @property
    def max_sparse_token(self):
        if self.sparse_state is None:
            return 0
        else:
            max_sparse_token = math.ceil(
                self.num_tokens * self.sparse_state.retention_ratio
            )
            # Ensure at least 1 token when sparse is enabled
            return max(1, max_sparse_token)

    @property
    def current_sparse_token(self):
        if self.sparse_state is None:
            return 0
        else:
            unuse_token_len = len(
                self.sparse_state.get_unuse_block_indices(
                    self.num_prompt_tokens, self.num_tokens
                )
            )
            n_keep = max(
                1, int(unuse_token_len * self.sparse_state.retention_ratio)
            )
            n_keep = min(n_keep, unuse_token_len)
            return n_keep

    @property
    def num_blocks(self):
        if self.sparse_state is None:
            return 0
        else:
            total_block_num = (
                self.max_sparse_token + self.block_size - 1
            ) // self.block_size
            return total_block_num

    @property
    def current_num_blocks(self):
        if self.sparse_state is None:
            return 0
        else:
            current_block_num = (
                self.current_sparse_token + self.block_size - 1
            ) // self.block_size
            return current_block_num

    @property
    def last_block_num_tokens_cached_sparse(self):
        # when current_sparse_token = n * block_size, it need return block_size
        if self.sparse_state.should_reuse_cache:
            return ((self.current_sparse_token - 1) % self.block_size) + 1
        elif self.sparse_state.should_compute_cache:
            return ((self.current_sparse_token - 1) % self.block_size) + 1
        raise NotImplementedError


    def __getstate__(self):
        return (
            self.model_type,
            self.num_tokens,
            self.num_prompt_tokens,
            self.token_ids,
            self.sampling_params,
            self.current_step,
            self.sparse_state,
        )

    def __setstate__(self, state):
        (
            self.model_type,
            self.num_tokens,
            self.num_prompt_tokens,
            self.token_ids,
            self.sampling_params,
            self.current_step,
            self.sparse_state,
        ) = state
