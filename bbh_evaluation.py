from vllm import LLM, SamplingParams
from src.bbh_evaluation import BBHEvaluation
import argparse


def parse_args():
    """
    Function to parse arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_model_path",
        type=str,
    )
    parser.add_argument(
        "--lora_path",
        type=str,
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--stopwords", nargs="*", default=[])
    parser.add_argument(
        "--min_cots",
        type=int,
        default=1,
        help="Min. number of CoTs you want to generate (min = 1)",
    )
    parser.add_argument(
        "--max_cots",
        type=int,
        default=1,
        help="Max. number of CoTs you want to generate (min = 1)",
    )
    parser.add_argument("--postprocess_responses", action="store_true")
    parser.add_argument("--chat_format", type=str, help="Options: llama_chat_simple, llama_chat_v2, llama_cot_chat, None")
    args = parser.parse_args()
    return args



def run_self_consistency(ARGS):
    sampling_params = SamplingParams(temperature=ARGS.temperature,
                                     max_tokens=ARGS.max_tokens,
                                     stop=ARGS.stopwords)
    enable_lora = ARGS.lora_path is not None
    llm = LLM(model=ARGS.base_model_path, enable_lora=enable_lora, max_lora_rank=64)
    benchmark = BenchmarkEvaluator(ARGS.split, k=ARGS.self_consistency_prompt_k, chat_format=ARGS.chat_format)
    if enable_lora:
        results = benchmark.self_consistency(llm,
                            sampling_params,
                            ARGS.lora_path,
                            postprocess_responses=ARGS.postprocess_responses,
                            self_consistency_k=ARGS.num_samples_self_consistency
                            )
    else:
        results = benchmark.self_consistency(llm,
                            sampling_params,
                            lora_path=None,
                            output_base_path=ARGS.base_model_path,
                            postprocess_responses=ARGS.postprocess_responses,
                            self_consistency_k=ARGS.num_samples_self_consistency)
    print(results)    


if __name__ == "__main__":
    print("Starting")
    ARGS = parse_args()


    sampling_params = SamplingParams(temperature=ARGS.temperature, max_tokens=ARGS.max_tokens, stop=ARGS.stopwords)
    enable_lora = ARGS.lora_path is not None
    llm = LLM(model=ARGS.base_model_path, enable_lora=enable_lora, max_lora_rank=64)
    for k in range(ARGS.min_cots, ARGS.max_cots+1):
        benchmark = BBHEvaluation(k=k)
        if enable_lora:
            results = benchmark(llm,
                                sampling_params,
                                ARGS.lora_path,
                                postprocess_responses=ARGS.postprocess_responses
                                )
        else:
            results = benchmark(llm,
                                sampling_params,
                                lora_path=None,
                                output_base_path=ARGS.base_model_path,
                                postprocess_responses=ARGS.postprocess_responses)
            
        print(f"Fininshed evaluation for k {k}")
        print(results)
