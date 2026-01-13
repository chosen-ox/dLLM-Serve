import os
import torch
from dataclasses import dataclass
from transformers import AutoConfig
from dllmserve.engine.sequence import ModelType


@dataclass
class Config:
    model: str
    model_type: ModelType = None
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.8
    tensor_parallel_size: int = 1
    torch_device: torch.device = None
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    max_logprobs: int = 1024
    log_optimize_level: int = 2

    def __post_init__(self):
        if self.torch_device is None:
            self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert os.path.isdir(self.model)
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(
            self.model, trust_remote_code=True, dtype=torch.bfloat16
        )
        model_type = getattr(self.hf_config, "model_type", None)
        if model_type in ["llada", "Dream"]:
            assert self.kvcache_block_size % 256 == 0
            self.model_type = ModelType.DIFFUSION
            # Handle different config attributes for LLaDA vs Dream
            if model_type == "llada":
                max_seq_len = self.hf_config.max_sequence_length
            else:  # Dream
                max_seq_len = self.hf_config.max_position_embeddings
            self.max_model_len = min(self.max_model_len, max_seq_len)
        else:
            raise ValueError(
                f"Unsupported model type: {model_type}. "
                f"Only 'llada' and 'Dream' diffusion models are supported."
            )
