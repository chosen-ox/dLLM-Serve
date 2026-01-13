import torch
from torch import nn
import torch.distributed as dist
from transformers import PretrainedConfig

from dllmserve.layers.attention import Attention
from dllmserve.layers.layernorm import RMSNorm
from dllmserve.layers.linear import ColumnParallelLinear, RowParallelLinear, MergedColumnParallelLinear
from dllmserve.layers.rotary_embedding import get_rope
from dllmserve.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from dllmserve.layers.activation import SiluAndMul


class DreamMLP(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x = self.down_proj(x)
        return x


class DreamAttention(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        tp_size = dist.get_world_size() if dist.is_initialized() else 1

        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        self.head_dim = self.hidden_size // self.total_num_heads
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.scaling = self.head_dim**-0.5

        # Dream uses bias=True for QKV projections
        self.q_proj = ColumnParallelLinear(
            self.hidden_size, self.total_num_heads * self.head_dim, bias=True
        )
        self.k_proj = ColumnParallelLinear(
            self.hidden_size, self.total_num_kv_heads * self.head_dim, bias=True
        )
        self.v_proj = ColumnParallelLinear(
            self.hidden_size, self.total_num_kv_heads * self.head_dim, bias=True
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=config.max_position_embeddings,
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

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for attention
        num_tokens = hidden_states.shape[0]
        q = q.view(num_tokens, self.num_heads, self.head_dim)
        k = k.view(num_tokens, self.num_kv_heads, self.head_dim)
        v = v.view(num_tokens, self.num_kv_heads, self.head_dim)

        # Apply rotary embeddings
        q, k = self.rotary_emb(positions, q, k)

        # Apply attention
        attn_output = self.attn(q, k, v)  # Returns (num_tokens, hidden_size)

        attn_output = self.o_proj(attn_output)
        return attn_output


class DreamDecoderLayer(nn.Module):
    def __init__(self, config: PretrainedConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = DreamAttention(config)
        self.mlp = DreamMLP(config)

        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self, positions: torch.Tensor, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        # Self-attention block
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states = residual + hidden_states

        # MLP block
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


class DreamForMaskedLM(nn.Module):
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.model = nn.Module()

        # Match checkpoint structure: model.embed_tokens
        self.model.embed_tokens = VocabParallelEmbedding(
            config.vocab_size, config.hidden_size
        )

        # Match checkpoint structure: model.layers
        self.model.layers = nn.ModuleList(
            [DreamDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )

        # Match checkpoint structure: model.norm
        self.model.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Match checkpoint structure: lm_head
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            bias=False,
        )

        for i, layer in enumerate(self.model.layers):
            layer.self_attn.attn.layer_id = i

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        # input_ids shape: (batch_size * seq_len,) - flattened
        # positions shape: (batch_size * seq_len,) - flattened

        hidden_states = self.model.embed_tokens(input_ids)
        # hidden_states shape: (batch_size * seq_len, hidden_size)

        for layer in self.model.layers:
            hidden_states = layer(positions, hidden_states)

        hidden_states = self.model.norm(hidden_states)
        return hidden_states

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Compute logits with DREAM's shifted prediction semantics.

        In DREAM: hidden_states[i] predicts token[i+1]
        This means logits[i] should be used to predict position i+1

        Args:
            hidden_states: [total_tokens, hidden_size]

        Returns:
            logits: [total_tokens, vocab_size]
                   logits[i] contains predictions for position i+1
        """
        return self.lm_head(hidden_states)

    @property
    def uses_shifted_prediction(self) -> bool:
        """Flag to indicate this model uses shifted prediction."""
        return True
