import os
from dllmserve import LLM, SamplingParams
from dllmserve.engine.sequence import ModelType
from transformers import AutoTokenizer


def main():
    # Use environment variable or download from HuggingFace
    path = os.environ.get("DREAM_MODEL_PATH", "Dream-org/Dream-v0-Instruct-7B")

    print("Initializing the LLM engine for Dream...")
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=1)

    print("Engine successfully configured for Diffusion Mode.")

    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)

    prompts = [
        "What is the Dream model and how does it differ from autoregressive models like GPT?",
        "Write a short, four-line poem about the quiet of a library at night.",
        "What is the capital of France?",
    ]

    list_of_sampling_params = [
        SamplingParams(
            temperature=0.0,
            gen_length=64,
            steps=64,
            cfg_scale=0.0,
            remasking="low_confidence",
        ),
        SamplingParams(
            temperature=0.0, gen_length=64, steps=64, cfg_scale=0.0, remasking="random"
        ),
        SamplingParams(
            temperature=0.0,
            gen_length=64,
            steps=64,
            cfg_scale=0.0,
            remasking="low_confidence",
        ),
    ]

    print("Applying chat template for Dream-Instruct...")
    formatted_prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        for p in prompts
    ]

    outputs = llm.generate(formatted_prompts, list_of_sampling_params, None)

    for i, output in enumerate(outputs):
        print("\n" + "=" * 80)
        print(f"Prompt {i+1}: {prompts[i]!r}")
        print(
            f"Params: gen_length={list_of_sampling_params[i].gen_length}, "
            f"steps={list_of_sampling_params[i].steps}, "
            f"cfg_scale={list_of_sampling_params[i].cfg_scale}"
        )
        print("-" * 80)
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
