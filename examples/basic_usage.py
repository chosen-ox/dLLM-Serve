import os
from dllmserve import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    # Use environment variable or download from HuggingFace
    path = os.environ.get("LLADA_MODEL_PATH", "llada-8b-instruct")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    # Use diffusion model sampling parameters
    sampling_params = SamplingParams(temperature=0.6, gen_length=64, steps=64, cfg_scale=0.0, remasking="low_confidence")
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
