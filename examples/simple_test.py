import os
from dllmserve import LLM, SamplingParams
from dllmserve.engine.sequence import ModelType
from transformers import AutoTokenizer

llm = LLM("./llada-instruct/", enforce_eager=True, tensor_parallel_size=1)

# Multiple prompts with same generation length will be batched together
prompts = ["Question 1?", "Question 2?", "Question 3?"]
sampling_params = SamplingParams(
    gen_length=128, steps=128, temperature=0.0, cfg_scale=0.0  # Same for all
)

outputs = llm.generate(prompts, sampling_params, None)


print(outputs)
