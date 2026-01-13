from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    # for diffusion llm
    gen_length: int = 128
    steps: int = 128
    cfg_scale: float = 0.0
    remasking: str = "low_confidence"

    def __post_init__(self):
        # Validate gen_length and steps are reasonable
        MAX_REASONABLE_GEN_LENGTH = 8192
        if self.gen_length > MAX_REASONABLE_GEN_LENGTH:
            raise ValueError(
                f"gen_length ({self.gen_length}) exceeds maximum reasonable value ({MAX_REASONABLE_GEN_LENGTH}). "
                f"This would cause excessive memory usage and slow generation."
            )
        if self.steps > MAX_REASONABLE_GEN_LENGTH:
            raise ValueError(
                f"steps ({self.steps}) exceeds maximum reasonable value ({MAX_REASONABLE_GEN_LENGTH}). "
                f"This would cause excessive computation time."
            )
        if self.gen_length <= 0:
            raise ValueError(f"gen_length must be positive, got {self.gen_length}")
        if self.steps <= 0:
            raise ValueError(f"steps must be positive, got {self.steps}")
