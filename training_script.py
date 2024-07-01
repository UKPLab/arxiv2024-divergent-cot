import argparse
import collections
import glob
import json
import math
import os
import random

import nltk
import torch
from datasets import load_dataset, Dataset
from peft import LoraConfig, PeftModel
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from tqdm import tqdm, trange
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    StoppingCriteria,
    StoppingCriteriaList,
    TrainingArguments,
    logging,
    pipeline,
)
from trl import SFTTrainer

from src.data_processors import DataProcessor, DataProcessorMode


def train(train_hf, tokenizer, ARGS):
    lora_r = 64
    lora_alpha = 16
    lora_dropout = 0.1
    use_4bit = True
    bnb_4bit_compute_dtype = "float16"
    bnb_4bit_quant_type = "nf4"
    use_nested_quant = False
    fp16 = False
    bf16 = False
    per_device_train_batch_size = ARGS.training_batch_size
    per_device_eval_batch_size = 1
    gradient_accumulation_steps = 1
    gradient_checkpointing = True
    max_grad_norm = 0.3
    learning_rate = 2e-4
    weight_decay = 0.001
    optim = "paged_adamw_32bit"
    lr_scheduler_type = "constant"
    max_steps = -1
    warmup_ratio = 0.03
    group_by_length = True
    max_seq_length = 4096
    packing = False

    # #Load Datasets

    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=use_4bit,
    #     bnb_4bit_quant_type=bnb_4bit_quant_type,
    #     bnb_4bit_compute_dtype=compute_dtype,
    #     bnb_4bit_use_double_quant=use_nested_quant,
    # )
    # model = AutoModelForCausalLM.from_pretrained(
    #     ARGS.base_model_path, quantization_config=bnb_config, device_map=device_map
    # )
    model = AutoModelForCausalLM.from_pretrained(
        ARGS.base_model_path,
        load_in_8bit=True,
        device_map="auto",
    )
    # torch_dtype=torch.float16,
    # model.config.use_cache = False
    # model.config.pretraining_tp = 1

    peft_config = LoraConfig(
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        r=lora_r,
        bias="none",
        task_type="CAUSAL_LM",
    )
    # Set training parameters
    training_arguments = TrainingArguments(
        output_dir=ARGS.lora_path,
        num_train_epochs=1,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        fp16=fp16,
        bf16=bf16,
        max_grad_norm=max_grad_norm,
        max_steps=max_steps,
        warmup_ratio=warmup_ratio,
        group_by_length=group_by_length,
        lr_scheduler_type=lr_scheduler_type,
        report_to="none",
        evaluation_strategy="no",
        save_strategy="steps",
        save_steps=int(len(train_hf)/ARGS.training_batch_size / ARGS.epochs),
        logging_strategy="no",
        gradient_checkpointing=gradient_checkpointing,
    )
    print("Save every ", int(len(train_hf)/ARGS.training_batch_size / ARGS.epochs), " steps")
    # Set supervised fine-tuning parameters
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_hf,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_arguments,
        packing=packing,
    )
    print("Training started...")
    trainer.train()
    trainer.model.save_pretrained(ARGS.lora_path)
    if ARGS.merge_weights:
        model = trainer.model.merge_and_unload()
        model.save_pretrained(os.path.join(ARGS.lora_path, "merged_model"))
        tokenizer.save_pretrained(os.path.join(ARGS.lora_path, "merged_model"))
    return trainer.model


def get_training_set(ARGS, eos_token):
    mode = None
    if ARGS.dcot:
        mode = DataProcessorMode.DCOT
    elif ARGS.cot:
        mode = DataProcessorMode.COT
    else:
        raise Exception("Need to set one of these modes: DCoT, CoT")
    
    dataset_processor = DataProcessor(
        ARGS.train_path, 
        mode=mode, 
        eos=eos_token, 
        epochs=ARGS.epochs, 
        seed=ARGS.seed, 
        chat_format=ARGS.chat_format, 
    )
    train_hf = dataset_processor.get_hf_dataset()
    return train_hf

def parse_args():
    """
    Function to parse arguments
    """
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--dev_path", type=str)
    # model
    parser.add_argument(
        "--base_model_path",
        type=str,
    )
    parser.add_argument(
        "--lora_path",
        type=str,
    )
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--training_batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--save_steps", type=int)
    parser.add_argument("--chat_format", type=str, help="Options: llama_chat_simple, llama_chat_v2, llama_cot_chat, None")
    parser.add_argument("--merge_weights", action="store_true")
    parser.add_argument("--k", type=int, help="Number of chains to generate for eval")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cot", action="store_true")
    parser.add_argument("--dcot", action="store_true", help="Divergent CoT")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    print("Starting")
    ARGS = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        ARGS.base_model_path, trust_remote_code=True
    )
    # llama and phi-2 does not include any pad token
    # padding should be on the left
    tokenizer.pad_token = "[PAD]"
    tokenizer.padding_side = "left"

    model = None
    if ARGS.train:
        train_hf = get_training_set(ARGS, tokenizer.eos_token)
        model = train(train_hf, tokenizer, ARGS)

