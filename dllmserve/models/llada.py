import torch
from torch import nn
import torch.distributed as dist
from transformers import PretrainedConfig

from dllmserve.layers.attention import Attention
from dllmserve.layers.layernorm import RMSNorm
from dllmserve.layers.linear import ColumnParallelLinear, RowParallelLinear
from dllmserve.layers.rotary_embedding import get_rope
from dllmserve.layers.embed_head import VocabParallelEmbedding, ParallelLMHead


class LLaDADecoderLayer(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        tp_size = dist.get_world_size() if dist.is_initialized() else 1

        self.hidden_size = config.d_model
        self.total_num_heads = config.n_heads
        self.total_num_kv_heads = getattr(config, "n_kv_heads", config.n_heads)
        self.head_dim = self.hidden_size // self.total_num_heads
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.scaling = self.head_dim**-0.5

        self.q_proj = ColumnParallelLinear(
            self.hidden_size, self.total_num_heads * self.head_dim, bias=False
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size, self.total_num_kv_heads * self.head_dim, bias=False
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size, self.total_num_kv_heads * self.head_dim, bias=False
        )
        self.attn_out = RowParallelLinear(
            self.total_num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self.ff_proj = ColumnParallelLinear(
            config.d_model, config.mlp_hidden_size, bias=False
        )
        self.up_proj = ColumnParallelLinear(
            config.d_model, config.mlp_hidden_size, bias=False
        )
        self.ff_out = RowParallelLinear(
            config.mlp_hidden_size, config.d_model, bias=False
        )
        self.act_fn = nn.SiLU()
        self.attn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ff_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_sequence_length,
            base=config.rope_theta,
        )
        self.attn = Attention(
            self.num_heads, self.head_dim, self.scaling, self.num_kv_heads, causal=False
        )

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # hidden_states shape: (total_tokens, hidden_size)
        # positions shape: (total_tokens,)

        residual = hidden_states

        normed_states = self.attn_norm(hidden_states)
        q = self.q_proj(normed_states)
        k = self.k_proj(normed_states)
        v = self.v_proj(normed_states)

        # Reshape for attention - already in correct shape
        num_tokens = hidden_states.shape[0]
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(num_tokens, self.num_kv_heads, self.head_dim)

        # Apply rotary embeddings
        q, k = self.rotary_emb(positions, q, k)

        # Apply attention
        attn_output = self.attn(q, k, v)  # Returns (num_tokens, hidden_size)

        attn_output = self.attn_out(attn_output)
        hidden_states = residual + attn_output

        # MLP block
        residual = hidden_states
        normed_states = self.ff_norm(hidden_states)
        ff_gate = self.act_fn(self.ff_proj(normed_states))
        ff_up = self.up_proj(normed_states)
        ff_output = self.ff_out(ff_gate * ff_up)
        hidden_states = residual + ff_output

        return hidden_states


class LLaDAForMaskedLM(nn.Module):
    packed_modules_mapping = {}

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.transformer = nn.Module()
        self.model.transformer.wte = VocabParallelEmbedding(
            config.vocab_size, config.d_model
        )
        self.model.transformer.blocks = nn.ModuleList(
            [LLaDADecoderLayer(config) for _ in range(config.n_layers)]
        )
        self.model.transformer.ln_f = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.model.transformer.ff_out = ParallelLMHead(
            getattr(config, "embedding_size", config.vocab_size),
            config.d_model,
            bias=False,
        )
        self.lm_head = self.model.transformer.ff_out

        for i, block in enumerate(self.model.transformer.blocks):
            block.attn.layer_id = i

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # input_ids shape: (batch_size * seq_len,) - flattened
        # positions shape: (batch_size * seq_len,) - flattened

        hidden_states = self.model.transformer.wte(input_ids)
        # hidden_states shape: (batch_size * seq_len, hidden_size)

        for layer in self.model.transformer.blocks:
            hidden_states = layer(positions, hidden_states)

        hidden_states = self.model.transformer.ln_f(hidden_states)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    @property
    def uses_shifted_prediction(self) -> bool:
        """LLaDA uses direct prediction (not shifted)."""
        return False
