"""Compare the opt-in Dream output path with the installed native attention backend.

Usage: python tests/performance/bench_final_layer_output.py --model /path/to/Dream
This measures the framework's complete generation, including host scheduling.
It does not enable CUDA Graphs or replace the installed attention implementation.
"""

import argparse
import json
import statistics
import time

import torch
from dllmserve import LLM, SamplingParams
from dllmserve.sparse.state import SparseConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--context", type=int, default=8192)
    parser.add_argument("--block", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    llm = LLM(
        args.model,
        max_num_seqs=1,
        max_model_len=args.context + args.block,
        max_num_batched_tokens=args.context + args.block,
        enforce_eager=True,
    )
    filler = llm.tokenizer.encode("Question: What is two plus two? Answer: Four.\n")
    ids = (filler * ((args.context + len(filler) - 1) // len(filler)))[: args.context]
    params = SamplingParams(temperature=0, gen_length=args.block, steps=args.block, ignore_eos=True)
    sparse = SparseConfig(enabled=True, default_block_len=args.block, head_select=True)
    reference = None
    values = {False: [], True: []}
    for rep in range(args.repeats + 1):
        for enabled in ((False, True) if rep % 2 == 0 else (True, False)):
            llm.config.optimize_final_layer = enabled
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = llm.generate([list(ids)], params, sparse_configs=sparse, use_tqdm=False)[0]
            torch.cuda.synchronize()
            seconds = time.perf_counter() - start
            if reference is None:
                reference = output["token_ids"]
            if output["token_ids"] != reference:
                raise RuntimeError("Generated tokens differ; inspect numerical/decision margins")
            if rep:
                values[enabled].append(seconds)
    baseline, optimized = (statistics.median(values[x]) for x in (False, True))
    print(
        json.dumps(
            {
                "baseline_seconds": baseline,
                "optimized_seconds": optimized,
                "time_reduction_percent": 100 * (1 - optimized / baseline),
                "tokens_match": True,
                "scope": "Native backend, framework generation",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
