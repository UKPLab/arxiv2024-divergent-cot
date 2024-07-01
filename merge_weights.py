import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

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
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    ARGS = parse_args()
    base_model = AutoModelForCausalLM.from_pretrained(
                ARGS.base_model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
    tokenizer = AutoTokenizer.from_pretrained(ARGS.base_model_path)

    lora_model = PeftModel.from_pretrained(base_model, ARGS.lora_path)
    model = lora_model.merge_and_unload()

    output_path = os.path.join(ARGS.lora_path, "merged_model")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)