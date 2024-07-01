from vllm import LLM, SamplingParams
from src.evaluation import *
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
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
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
    parser.add_argument("--do_self_consistency", action="store_true")
    parser.add_argument("--num_samples_self_consistency", type=int, default=5)
    parser.add_argument("--self_consistency_prompt_k", type=int, default=1)
    parser.add_argument("--tasks2k_path", type=str, help="Path to the json file that specifies the best number of CoTs (k) per task.")
    parser.add_argument("--dcot_self_consistency",  action="store_true")
    args = parser.parse_args()
    return args


def run_benchmark(llm, sampling_params, enable_lora, ARGS):
    if enable_lora:
        results = benchmark(llm, sampling_params, ARGS.lora_path)
    else:
        results = benchmark(
            llm,
            sampling_params,
            lora_path=None,
            output_base_path=ARGS.base_model_path,
            chat_format=ARGS.chat_format
        )
    return results

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
                            self_consistency_k=ARGS.num_samples_self_consistency
                            )
    print(results)    


if __name__ == "__main__":
    print("Starting")
    ARGS = parse_args()

    if ARGS.do_self_consistency:
        run_self_consistency(ARGS)
    else:
        sampling_params = SamplingParams(temperature=ARGS.temperature, max_tokens=ARGS.max_tokens, stop=ARGS.stopwords)
        enable_lora = ARGS.lora_path is not None
        llm = LLM(model=ARGS.base_model_path, enable_lora=enable_lora, max_lora_rank=64)
        if ARGS.split == "test" and ARGS.tasks2k_path is not None:
            with open(ARGS.tasks2k_path) as f:
                tasks2k = json.load(f)
            print(tasks2k)
            benchmark = BenchmarkEvaluator(ARGS.split, k=1, chat_format=ARGS.chat_format)
            if enable_lora:
                results = benchmark.test_set_eval(tasks2k, 
                                        ARGS.chat_format, 
                                        llm,
                                        sampling_params,
                                        ARGS.lora_path,
                                        postprocess_responses=ARGS.postprocess_responses,
                                        self_consistency=ARGS.dcot_self_consistency,
                                        num_samples_self_consistency=ARGS.num_samples_self_consistency)
            else:
                results = benchmark.test_set_eval(tasks2k, 
                                        ARGS.chat_format, 
                                        llm,
                                        sampling_params,
                                        lora_path=None,
                                        output_base_path=ARGS.base_model_path,
                                        postprocess_responses=ARGS.postprocess_responses,
                                        self_consistency=ARGS.dcot_self_consistency,
                                        num_samples_self_consistency=ARGS.num_samples_self_consistency)
            print(results)
        else:
            for k in range(ARGS.min_cots, ARGS.max_cots+1):
                benchmark = BenchmarkEvaluator(ARGS.split, k=k, chat_format=ARGS.chat_format)
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
