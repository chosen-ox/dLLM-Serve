import pickle
import torch
import torch.distributed as dist
import torch.nn.functional as F
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
from typing import Optional
import numpy as np
import math
import os

from dllmserve.config import Config
from dllmserve.engine.sequence import Sequence
from dllmserve.sampling_params import SamplingParams
from dllmserve.utils.context import set_context, get_context, reset_context
from dllmserve.utils.loader import load_model
from dllmserve.models.llada import LLaDAForMaskedLM
from dllmserve.models.dream import DreamForMaskedLM
from dllmserve.engine.sequence import ModelType
from dllmserve.sparse.state import (
    SparseConfig,
    PerRequestSparseState,
    SparseContext,
)


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        self.model_type = config.model_type
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        master_port = os.environ.get("MASTER_PORT", "2333")

        if not dist.is_initialized():
            dist.init_process_group(
                "nccl",
                f"tcp://localhost:{master_port}",
                world_size=self.world_size,
                rank=rank,
            )
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()

        hf_config.dtype = torch.bfloat16
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device(config.torch_device)

        # Initialize diffusion model (LLaDA or Dream)
        if getattr(hf_config, "model_type", None) == "Dream":
            self.model = DreamForMaskedLM(hf_config)
        else:  # LLaDA or other diffusion models
            self.model = LLaDAForMaskedLM(hf_config)

        load_model(self.model, config.model)

        # Diffusion model warmup and cache allocation
        self.warmup_model_diffusion()
        self.allocate_kv_cache()

        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="dllmserve", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="dllmserve")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4 : n + 4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4 : n + 4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model_diffusion(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        free, total = torch.cuda.mem_get_info()

        print("GPU memory (GB): free", free / (1024**3), "total", total / (1024**3))

        max_num_batched_tokens, max_model_len = (
            self.config.max_num_batched_tokens,
            self.config.max_model_len,
        )
        num_seqs = min(
            max_num_batched_tokens // max_model_len, self.config.max_num_seqs
        )

        prompt_len = (
            max_model_len - 1
            if max_model_len - SamplingParams.gen_length <= 0
            else max_model_len - SamplingParams.gen_length
        )
        seqs = [
            Sequence(
                [0] * prompt_len,
                SamplingParams(
                    temperature=0, gen_length=max_model_len - prompt_len, steps=1
                ),
                self.config,
                None,
            )
            for _ in range(num_seqs)
        ]
        self.run(seqs, None)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        # Handle different config attributes for LLaDA vs Dream
        hf_model_type = getattr(config.hf_config, "model_type", None)
        if hf_model_type == "Dream":
            num_layers = hf_config.num_hidden_layers
            num_heads = hf_config.num_attention_heads
            num_kv_heads = hf_config.num_key_value_heads
            hidden_size = hf_config.hidden_size
        else:  # LLaDA
            num_layers = hf_config.n_layers
            num_heads = hf_config.n_heads
            num_kv_heads = hf_config.n_kv_heads
            hidden_size = hf_config.d_model

        head_dim = hidden_size // num_heads
        num_kv_heads = num_kv_heads // self.world_size
        block_bytes = (
            2
            * num_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * torch.bfloat16.itemsize
        )

        config.num_kvcache_blocks = (
            int(total * config.gpu_memory_utilization - used - peak + current)
            // block_bytes
        )

        print(f"config.num_kvcache_blocks: {config.num_kvcache_blocks}")

        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.zeros(
            2,
            num_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
        )

        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table))
            for seq in seqs
            if seq.sparse_state is not None
        ]
        block_tables = torch.tensor(
            block_tables, dtype=torch.int32, pin_memory=True
        ).cuda(non_blocking=True)
        return block_tables

    @torch.inference_mode()
    def _run_diffusion(self, seqs: list[Sequence]) -> list[list[int]] | None:
        """
        Run diffusion model inference with FlashAttention varlen batching.
        """
        if not seqs:
            return [] if self.rank == 0 else None

        batch_size = len(seqs)
        device = "cuda"

        k_cu_seqlens_compute = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=device
        )
        k_lengths = torch.zeros(batch_size, dtype=torch.int32, device=device)

        input_ids_list = []
        slot_mapping = []
        extra_slot_mapping = []
        position_ids_list = []
        for i, seq in enumerate(seqs):
            input_tokens = torch.tensor(seq.token_ids, dtype=torch.int64, device=device)
            if (
                seq.sparse_state is None
                or seq.sparse_state.is_full_attention
                or seq.sparse_state.should_compute_cache
            ):
                # Need all tokens for this sequence
                k_lengths[i] = seq.num_tokens
                input_ids_list.append(input_tokens)
                position_ids_list.append(
                    torch.arange(0, seq.num_tokens, dtype=torch.int64, device=device)
                )

                if seq.sparse_state is None or seq.sparse_state.is_full_attention:
                    slot_mapping.append([])
                else:
                    current_slot_mapping = []
                    for j in range(seq.num_should_cached_blocks):
                        start = seq.block_table[j] * self.block_size
                        if j != seq.num_should_cached_blocks - 1:
                            end = start + self.block_size
                        else:
                            end = start + seq.last_block_num_tokens_cached_sparse
                        current_slot_mapping.extend(list(range(start, end)))

                    slot_mapping.append(
                        torch.tensor(
                            current_slot_mapping,
                            dtype=torch.int32,
                            pin_memory=True,
                            device=device,
                        )
                    )

            elif seq.sparse_state.should_reuse_cache:
                # Only need current block
                block_indices = seq.sparse_state.get_block_indices(
                    seq.num_prompt_tokens, seq.num_tokens
                )

                k_lengths[i] = len(block_indices)
                input_ids_list.append(input_tokens[block_indices])
                position_ids_list.append(block_indices)

                current_slot_mapping = []

                for j in range(seq.num_should_cached_blocks):
                    start = seq.block_table[j] * self.block_size
                    if j != seq.num_should_cached_blocks - 1:
                        end = start + self.block_size
                    else:
                        end = start + seq.last_block_num_tokens_cached_sparse
                    current_slot_mapping.extend(list(range(start, end)))

                slot_mapping.append(
                    torch.tensor(
                        current_slot_mapping,
                        dtype=torch.int32,
                        pin_memory=True,
                        device=device,
                    )
                )

        k_cu_seqlens_compute[1:] = torch.cumsum(k_lengths, dim=0)
        input_ids_compute = torch.cat(input_ids_list, dim=0)
        position_ids_compute = torch.cat(position_ids_list, dim=0)
        block_tables = self.prepare_block_tables(seqs)

        # Set context with optimized indices
        set_context(
            is_prefill=True,
            cu_seqlens_q=None,
            cu_seqlens_k=k_cu_seqlens_compute,
            max_seqlen_q=0,
            max_seqlen_k=max(k_cu_seqlens_compute),
            is_diffusion=True,
            seqs=seqs,
            slot_mapping=slot_mapping,
            extra_slot_mapping=extra_slot_mapping,
            block_tables=block_tables,
        )

        # Forward through model with only necessary tokens
        hidden = self.model(input_ids_compute, position_ids_compute)

        # For Dream shifted prediction: shift hidden states so hidden[i] predicts token[i]
        # Original: hidden[i] predicts token[i+1]
        # After shift: shifted_hidden[i] = hidden[i-1] predicts token[i]
        if hasattr(self.model, 'uses_shifted_prediction') and self.model.uses_shifted_prediction:
            shifted_hidden_list = []
            for seq_idx in range(len(seqs)):
                seq_start = k_cu_seqlens_compute[seq_idx]
                seq_end = k_cu_seqlens_compute[seq_idx + 1]
                seq_hidden = hidden[seq_start:seq_end]

                # Shift: [h0, h1, h2, ...] -> [h0, h0, h1, h2, ...]
                shifted_seq_hidden = torch.cat([seq_hidden[:1], seq_hidden[:-1]], dim=0)
                shifted_hidden_list.append(shifted_seq_hidden)

            hidden = torch.cat(shifted_hidden_list, dim=0)

        if self.config.log_optimize_level != 2:
            cond_logits_flat = self.model.compute_logits(hidden)
        reset_context()

        # Handle CFG if needed (similar optimization)
        cfg_scales = torch.tensor(
            [s.sampling_params.cfg_scale for s in seqs], device=device
        )
        need_cfg = bool((cfg_scales > 0).any())

        uncond_logits_flat = None
        if need_cfg:
            raise NotImplementedError("CFG is not implemented!")

        if self.config.log_optimize_level == 2:
            out = self._reconstruct_and_sample_optimized_v2(
                hidden,
                uncond_logits_flat,
                seqs,
                k_cu_seqlens_compute,
                device,
            )
        elif self.config.log_optimize_level == 1:
            out = self._reconstruct_and_sample_optimized(
                cond_logits_flat,
                uncond_logits_flat,
                seqs,
                k_cu_seqlens_compute,
                device,
            )
        else:
            out = self._reconstruct_and_sample(
                cond_logits_flat,
                uncond_logits_flat,
                seqs,
                k_cu_seqlens_compute,
                device,
            )

        # print("after constructed")
        # print("PyTorch reserved:", torch.cuda.memory_reserved()/(1024**3), "GB")
        # print("PyTorch allocated:", torch.cuda.memory_allocated()/(1024**3), "GB")
        return out

    def _reconstruct_and_sample(
        self,
        cond_logits_flat: torch.Tensor,
        uncond_logits_flat: Optional[torch.Tensor],
        seqs: list[Sequence],
        k_cu_seqlens_compute: torch.Tensor,
        device: torch.device,
    ) -> Optional[list[list[int]]]:
        """
        Reconstruct full sequence outputs and sample next tokens.
        For sparse sequences, only unmask tokens in current block.
        """
        if self.rank != 0:
            return None

        batch_size = len(seqs)
        lengths = torch.tensor(
            [len(s.token_ids) for s in seqs], dtype=torch.int32, device=device
        )
        max_len = lengths.max().item()
        vocab_size = cond_logits_flat.shape[-1]
        mask_token = self.config.hf_config.mask_token_id

        # Initialize full logits with -inf (for padding)
        cond_logits_full = torch.full(
            (batch_size, max_len, vocab_size),
            -float("inf"),
            device=device,
            dtype=cond_logits_flat.dtype,
        )

        # Scatter computed logits back to full positions
        for seq_idx in range(batch_size):
            seq = seqs[seq_idx]
            k_seq_start, k_seq_end = (
                k_cu_seqlens_compute[seq_idx],
                k_cu_seqlens_compute[seq_idx + 1],
            )
            if seq.sparse_state is None or seq.sparse_state.is_full_attention:
                # Full attention: use generation positions only
                seq_positions = torch.arange(
                    seq.num_prompt_tokens, seq.num_tokens, dtype=torch.int64, device=device
                )
                # After shift, extract only generation logits (skip prompt logits)
                cond_logits_full[seq_idx, seq_positions] = cond_logits_flat[
                    k_seq_start + seq.num_prompt_tokens : k_seq_end
                ]
            elif seq.sparse_state.should_compute_cache:
                # Computing cache: use block indices
                seq_positions = seq.sparse_state.get_block_indices(
                    seq.num_prompt_tokens, seq.num_tokens
                )
                # Extract logits for block positions (not from start)
                block_start = seq_positions[0].item()
                block_end = seq_positions[-1].item() + 1
                cond_logits_full[seq_idx, seq_positions] = cond_logits_flat[
                    k_seq_start + block_start : k_seq_start + block_end
                ]
            elif seq.sparse_state.should_reuse_cache:
                # Reusing cache: use block indices
                seq_positions = seq.sparse_state.get_block_indices(
                    seq.num_prompt_tokens, seq.num_tokens
                )
                cond_logits_full[seq_idx, seq_positions] = cond_logits_flat[
                    k_seq_start:k_seq_end
                ]

        logits = cond_logits_full

        # Temperature sampling
        temps = torch.tensor(
            [s.sampling_params.temperature for s in seqs], device=device
        )
        if not (temps == 0).all():
            gumbel = -torch.log(
                -torch.log(torch.rand_like(logits, dtype=torch.float32))
            )
            logits = logits / temps.view(batch_size, 1, 1).clamp(min=1e-8) + gumbel

        # Get predicted tokens
        # [B, max_len]
        x0 = torch.argmax(logits, dim=-1)

        # Create mask for current tokens
        # Pad token - use config's pad_token_id
        pad_token = self.config.hf_config.pad_token_id
        input_ids_pad = torch.full(
            (batch_size, max_len), pad_token, dtype=torch.int64, device=device
        )

        for i in range(batch_size):
            seq = seqs[i]
            seq_len = seq.num_tokens
            input_ids_pad[i, :seq_len] = torch.tensor(
                seqs[i].token_ids, dtype=torch.int64, device=device
            )

        # Generated part which is masked
        mask_indices = input_ids_pad == mask_token
        # Position to unmask
        unmask_masks = torch.zeros_like(mask_indices, dtype=torch.bool)

        for i, seq in enumerate(seqs):
            if seq.sparse_state is None:
                # No sparse config - use full attention logic for all positions
                unmask_masks[i, seq.num_prompt_tokens : seq.num_tokens] = mask_indices[
                    i, seq.num_prompt_tokens : seq.num_tokens
                ]
            elif (
                seq.sparse_state.is_full_attention
                or seq.sparse_state.should_compute_cache
                or seq.sparse_state.should_reuse_cache
            ):
                # After shift, predictions work like LLaDA
                block_indices = seq.sparse_state.get_block_indices(
                    seq.num_prompt_tokens, seq.num_tokens
                )
                unmask_masks[i, block_indices] = mask_indices[i, block_indices]

        # Compute confidence scores (after shift, works like LLaDA)
        probs = torch.softmax(logits, dim=-1)
        x0_probs = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)
        confidence = torch.where(unmask_masks, x0_probs, -float("inf"))

        # Apply remasking strategy
        for i, seq in enumerate(seqs):
            if seq.sampling_params.remasking == "random" and unmask_masks[i].any():
                nmask = unmask_masks[i].sum().item()
                confidence[i, unmask_masks[i]] = torch.rand(nmask, device=device, dtype=confidence.dtype)

        # Calculate how many tokens to unmask per sequence
        steps_remain = torch.zeros(batch_size, device=device, dtype=torch.float32)

        for i in range(batch_size):
            seq = seqs[i]
            if seq.sparse_state is None:
                steps_remain[i] = seq.sampling_params.steps - seq.current_step
            elif (
                seq.sparse_state.is_full_attention
                or seq.sparse_state.should_compute_cache
                or seq.sparse_state.should_reuse_cache
            ):
                steps_remain[i] = (
                    seq.sparse_state.compute_block_boundaries(
                        seq.num_prompt_tokens, seq.num_tokens
                    )[1]
                    - seq.sparse_state.step_in_block
                )

        # Only count masks we're considering
        num_masks = unmask_masks.sum(dim=1)
        num_to_unmask = torch.ceil(num_masks.float() / steps_remain).long()
        num_to_unmask = torch.minimum(num_to_unmask, num_masks)

        # Update sequences
        updated_ids = input_ids_pad.clone()
        for i in range(batch_size):
            if num_to_unmask[i] > 0 and unmask_masks[i].any():
                seq = seqs[i]

                # Special handling for is_full_attention: first token of block
                if seq.sparse_state is not None and seq.sparse_state.is_full_attention:
                    block_indices = seq.sparse_state.get_block_indices(
                        seq.num_prompt_tokens, seq.num_tokens
                    )
                    if len(block_indices) > 0:
                        first_token_pos = block_indices[0].item()

                        # Check if first token is still masked
                        if input_ids_pad[i, first_token_pos] == mask_token:
                            # Unmask first token directly (counts as 1 token)
                            updated_ids[i, first_token_pos] = x0[i, first_token_pos]

                            # If need to unmask more tokens, use confidence-based selection
                            if num_to_unmask[i] > 1:
                                # Exclude first token from confidence-based selection
                                remaining_unmask_mask = unmask_masks[i].clone()
                                remaining_unmask_mask[first_token_pos] = False

                                if remaining_unmask_mask.any():
                                    masked_conf = confidence[i, remaining_unmask_mask]
                                    num_remaining = min(num_to_unmask[i].item() - 1, remaining_unmask_mask.sum().item())
                                    _, topk = torch.topk(masked_conf, k=num_remaining)
                                    masked_positions = torch.where(remaining_unmask_mask)[0]
                                    top_indices = masked_positions[topk]
                                    updated_ids[i, top_indices] = x0[i, top_indices]
                            continue

                # Normal handling (first token already unmasked, or not is_full_attention)
                masked_conf = confidence[i, unmask_masks[i]]
                _, topk = torch.topk(masked_conf, k=num_to_unmask[i].item())
                masked_positions = torch.where(unmask_masks[i])[0]
                top_indices = masked_positions[topk]
                updated_ids[i, top_indices] = x0[i, top_indices]

        # Return updated sequences
        result = []
        for i, seq in enumerate(seqs):
            result.append(updated_ids[i, : seq.num_tokens].tolist())

        return result

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        return self._run_diffusion(seqs)

    # TODO: should not be a class function
    def add_gumbel_noise(
        self, logits: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        if temperature == 0:
            return logits
        logits = logits.to(torch.float64)
        noise = torch.rand_like(logits, dtype=torch.float64)
        gumbel_noise = (-torch.log(noise)) ** temperature
        return logits.exp() / gumbel_noise

    def _reconstruct_and_sample_optimized(
        self,
        cond_logits_flat: torch.Tensor,
        uncond_logits_flat: Optional[torch.Tensor],
        seqs: list[Sequence],
        k_cu_seqlens_compute: torch.Tensor,
        device: torch.device,
    ) -> Optional[list[list[int]]]:
        if self.rank != 0:
            return None

        batch_size = len(seqs)

        # mask_token = 126336
        mask_token = self.config.hf_config.mask_token_id

        # logits pos and len
        seq_logits_list = []
        seq_indices_list = []
        seq_lengths = []

        for seq_idx in range(batch_size):
            seq = seqs[seq_idx]
            k_seq_start, k_seq_end = (
                k_cu_seqlens_compute[seq_idx],
                k_cu_seqlens_compute[seq_idx + 1],
            )

            if seq.sparse_state is None:
                gen_start = seq.num_prompt_tokens
                seq_logits = cond_logits_flat[k_seq_start + gen_start : k_seq_end]
                indices = torch.arange(
                    gen_start, seq.num_tokens, dtype=torch.int64, device=device
                )
            elif seq.sparse_state.should_reuse_cache:
                seq_logits = cond_logits_flat[k_seq_start:k_seq_end]
                indices = seq.sparse_state.get_block_indices(
                    seq.num_prompt_tokens, seq.num_tokens
                )
            elif (
                seq.sparse_state.is_full_attention
                or seq.sparse_state.should_compute_cache
            ):
                indices = seq.sparse_state.get_block_indices(
                    seq.num_prompt_tokens, seq.num_tokens
                )
                # Extract logits for block positions (not from start)
                block_start = indices[0].item()
                block_end = indices[-1].item() + 1
                seq_logits = cond_logits_flat[k_seq_start + block_start : k_seq_start + block_end]

            seq_logits_list.append(seq_logits)
            seq_indices_list.append(indices)
            seq_lengths.append(len(seq_logits))

        # add gumbel
        for seq_idx in range(batch_size):
            seq = seqs[seq_idx]
            temperature = seq.sampling_params.temperature
            if temperature > 0:
                seq_logits_list[seq_idx] = self.add_gumbel_noise(
                    seq_logits_list[seq_idx], temperature
                )

        # max seq len
        max_valid_len = max(seq_lengths)
        vocab_size = cond_logits_flat.shape[-1]

        batch_logits = torch.full(
            (batch_size, max_valid_len, vocab_size),
            -float("inf"),
            device=device,
            dtype=cond_logits_flat.dtype,
        )

        for seq_idx in range(batch_size):
            seq_len = seq_lengths[seq_idx]
            batch_logits[seq_idx, :seq_len] = seq_logits_list[seq_idx]

        # x0 = torch.argmax(batch_logits, dim=-1)  # [bs, max_len]

        if seq.sampling_params.remasking == "random":
            x0 = torch.argmax(batch_logits, dim=-1)  # [bs, max_len]
            x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
            x0_p = torch.rand(
                (batch_logits.shape[0], batch_logits.shape[1]), device=x0.device
            )

        else:  # low_confidence
            probs = F.softmax(batch_logits, dim=-1)  # [bs, max_len, vocab]
            # [bs, max_len, 1] -> [bs, max_len]
            # x0_p = probs.gather(-1, x0.unsqueeze(-1)).squeeze(-1)  # [bs, max_len]
            x0_p, x0 = torch.max(probs, dim=-1)

        result = []
        for seq_idx in range(batch_size):
            seq = seqs[seq_idx]
            indices = seq_indices_list[seq_idx]
            seq_len = seq_lengths[seq_idx]
            seq_x0 = x0[seq_idx, :seq_len]  # [seq_len]
            seq_x0_p = x0_p[seq_idx, :seq_len]  # [seq_len]
            token_ids = torch.tensor(seq.token_ids, dtype=torch.int64, device=device)

            # After shift, indices[i] predicts token[i] (like LLaDA)
            mask_index = token_ids[indices] == mask_token
            confidence = torch.where(mask_index, seq_x0_p, -np.inf)

            if seq.sparse_state is None:
                steps_remain = seq.sampling_params.steps - seq.current_step
            elif (
                seq.sparse_state.is_full_attention
                or seq.sparse_state.should_compute_cache
                or seq.sparse_state.should_reuse_cache
            ):
                steps_remain = (
                    seq.sparse_state.compute_block_boundaries(
                        seq.num_prompt_tokens, seq.num_tokens
                    )[1]
                    - seq.sparse_state.step_in_block
                )
            nmask = mask_index.sum().item()
            num_to_unmask = min(math.ceil(float(nmask) / steps_remain), nmask)

            # Special handling for is_full_attention: always unmask first token of block
            if seq.sparse_state is not None and seq.sparse_state.is_full_attention and len(indices) > 0 and num_to_unmask > 0:
                first_token_pos = indices[0].item()

                # Check if first token is still masked
                if token_ids[first_token_pos] == mask_token:
                    # Unmask first token directly (counts as 1 token)
                    token_ids[first_token_pos] = seq_x0[0]

                    # If need to unmask more tokens, use confidence-based selection
                    if num_to_unmask > 1:
                        # Exclude first token from confidence-based selection
                        remaining_confidence = confidence.clone()
                        remaining_confidence[0] = -np.inf

                        num_remaining = min(num_to_unmask - 1, (confidence > -np.inf).sum().item() - 1)
                        if num_remaining > 0:
                            _, topk_remaining = torch.topk(remaining_confidence, k=num_remaining)
                            token_ids[indices[topk_remaining]] = seq_x0[topk_remaining]

                    result.append(token_ids.tolist())
                    continue

            # Normal handling (first token already unmasked, or not is_full_attention)
            _, topk = torch.topk(confidence, k=num_to_unmask)
            token_ids[indices[topk]] = seq_x0[topk]

            result.append(token_ids.tolist())

        return result

    def _reconstruct_and_sample_optimized_v2(
        self,
        hidden_states: torch.Tensor,
        uncond_logits_flat: Optional[torch.Tensor],
        seqs: list[Sequence],
        k_cu_seqlens_compute: torch.Tensor,
        device: torch.device,
    ) -> Optional[list[list[int]]]:
        if self.rank != 0:
            return None

        batch_size = len(seqs)

        # mask_token = 126336
        mask_token = self.config.hf_config.mask_token_id

        # logits pos and len
        seq_hidden_list = []
        seq_indices_list = []
        seq_lengths = []

        seq_lengths = torch.zeros(batch_size, dtype=torch.int32, device=device)

        cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=device)
        max_logprobs = self.config.max_logprobs
        cond_logits_flat = torch.zeros(
            (max_logprobs, self.config.hf_config.vocab_size), device=device
        )
        x0_list = []
        x0_p_list = []

        cur_num_logprobs = 0
        last_seq_idx = 0

        for seq_idx in range(batch_size):
            seq = seqs[seq_idx]
            k_seq_start, k_seq_end = (
                k_cu_seqlens_compute[seq_idx],
                k_cu_seqlens_compute[seq_idx + 1],
            )

            if seq.sparse_state is None:
                gen_start = seq.num_prompt_tokens
                seq_hidden = hidden_states[k_seq_start + gen_start : k_seq_end]
                indices = torch.arange(
                    gen_start, seq.num_tokens, dtype=torch.int64, device=device
                )
            elif seq.sparse_state.should_reuse_cache:
                seq_hidden = hidden_states[k_seq_start:k_seq_end]
                indices = seq.sparse_state.get_block_indices(
                    seq.num_prompt_tokens, seq.num_tokens
                )
            elif (
                seq.sparse_state.is_full_attention
                or seq.sparse_state.should_compute_cache
            ):
                indices = seq.sparse_state.get_block_indices(
                    seq.num_prompt_tokens, seq.num_tokens
                )
                # Extract hidden states for block positions (not from start)
                block_start = indices[0].item()
                block_end = indices[-1].item() + 1
                seq_hidden = hidden_states[k_seq_start + block_start : k_seq_start + block_end]

            # cur_num_logprobs += len(seq_logits)
            # TODO: handle situation that single seq length > max_logprobs
            if (
                cur_num_logprobs + len(seq_hidden) > max_logprobs
                and cur_num_logprobs > 0
            ):
                seq_hidden_tensor = torch.cat(seq_hidden_list, dim=0)
                cond_logits_flat = self.model.compute_logits(seq_hidden_tensor)
                cu_seqlens[1:] = torch.cumsum(seq_lengths[last_seq_idx], dim=0)
                for i in range(seq_idx - last_seq_idx):
                    seq_start, seq_end = (cu_seqlens[i], cu_seqlens[i + 1])
                    seq = seqs[i + last_seq_idx]
                    temperature = seq.sampling_params.temperature
                    if temperature > 0:
                        cond_logits_flat[seq_start:seq_end] = self.add_gumbel_noise(
                            cond_logits_flat[seq_start:seq_end], temperature
                        )

                probs = F.softmax(cond_logits_flat, dim=-1)  # [num_tokens, vocab]
                x0_p, x0 = torch.max(probs, dim=-1)
                x0_list.append(x0)
                x0_p_list.append(x0_p)

                last_seq_idx = seq_idx
                del cond_logits_flat
                seq_hidden_list.clear()
                cur_num_logprobs = 0

            cur_num_logprobs += len(seq_hidden)

            seq_hidden_list.append(seq_hidden)
            seq_indices_list.append(indices)
            seq_lengths[seq_idx] = len(seq_hidden)

        if cur_num_logprobs > 0:
            seq_idx = batch_size - 1
            seq_hidden_tensor = torch.cat(seq_hidden_list, dim=0)
            cond_logits_flat = self.model.compute_logits(seq_hidden_tensor)
            cu_seqlens[1:] = torch.cumsum(seq_lengths[last_seq_idx], dim=0)
            for i in range(seq_idx - last_seq_idx + 1):
                seq_start, seq_end = (cu_seqlens[i], cu_seqlens[i + 1])
                seq = seqs[i + last_seq_idx]
                temperature = seq.sampling_params.temperature
                if temperature > 0:
                    cond_logits_flat[seq_start:seq_end] = self.add_gumbel_noise(
                        cond_logits_flat[seq_start:seq_end], temperature
                    )

            probs = F.softmax(cond_logits_flat, dim=-1)  # [num_tokens, vocab]
            x0_p, x0 = torch.max(probs, dim=-1)

            x0_list.append(x0)
            x0_p_list.append(x0_p)

            del cond_logits_flat

        cu_seqlens[1:] = torch.cumsum(seq_lengths, dim=0)
        x0 = torch.cat(x0_list, dim=0)
        x0_p = torch.cat(x0_p_list, dim=0)

        result = []
        for seq_idx in range(batch_size):
            seq = seqs[seq_idx]
            seq_start, seq_end = (cu_seqlens[seq_idx], cu_seqlens[seq_idx + 1])
            indices = seq_indices_list[seq_idx]
            seq_len = seq_lengths[seq_idx]

            seq_x0 = x0[seq_start:seq_end]  # [seq_len]
            seq_x0_p = x0_p[seq_start:seq_end]  # [seq_len]
            token_ids = torch.tensor(seq.token_ids, dtype=torch.int64, device=device)

            # After shift, indices[i] predicts token[i] (like LLaDA)
            mask_index = token_ids[indices] == mask_token
            seq_x0_p_for_unmask = seq_x0_p

            if seq.sampling_params.remasking == "random":
                seq_x0_p_for_unmask = seq_x0_p_for_unmask.clone()
                seq_x0_p_for_unmask.uniform_(0, 1)
            elif seq.sampling_params.remasking == "low_confidence":
                pass
            else:
                raise NotImplementedError(seq.sampling_params.remasking)
            confidence = torch.where(mask_index, seq_x0_p_for_unmask, -np.inf)

            if seq.sparse_state is None:
                steps_remain = seq.sampling_params.steps - seq.current_step
            elif (
                seq.sparse_state.is_full_attention
                or seq.sparse_state.should_compute_cache
                or seq.sparse_state.should_reuse_cache
            ):
                steps_remain = (
                    seq.sparse_state.compute_block_boundaries(
                        seq.num_prompt_tokens, seq.num_tokens
                    )[1]
                    - seq.sparse_state.step_in_block
                )
            nmask = mask_index.sum().item()
            num_to_unmask = min(math.ceil(float(nmask) / steps_remain), nmask)

            # Special handling for is_full_attention: always unmask first token of block
            if seq.sparse_state is not None and seq.sparse_state.is_full_attention and len(indices) > 0 and num_to_unmask > 0:
                first_token_pos = indices[0].item()

                # Check if first token is still masked
                if token_ids[first_token_pos] == mask_token:
                    # Unmask first token directly (counts as 1 token)
                    token_ids[first_token_pos] = seq_x0[0]

                    # If need to unmask more tokens, use confidence-based selection
                    if num_to_unmask > 1:
                        # Exclude first token from confidence-based selection
                        remaining_confidence = confidence.clone()
                        remaining_confidence[0] = -np.inf

                        num_remaining = min(num_to_unmask - 1, (confidence > -np.inf).sum().item() - 1)
                        if num_remaining > 0:
                            _, topk_remaining = torch.topk(remaining_confidence, k=num_remaining)
                            token_ids[indices[topk_remaining]] = seq_x0[topk_remaining]

                    result.append(token_ids.tolist())
                    continue

            # Normal handling (first token already unmasked, or not is_full_attention)
            _, topk = torch.topk(confidence, k=num_to_unmask)
            token_ids[indices[topk]] = seq_x0[topk]

            result.append(token_ids.tolist())

        return result
