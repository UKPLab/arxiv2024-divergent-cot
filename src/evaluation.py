import collections
import json
import os
import re
import subprocess as sp
import time

import nltk
import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from evaluate import load
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm, trange
# from vllm.lora.request import LoRARequest

from .data_processors import Prompt


def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = (
        sp.check_output(command.split()).decode("ascii").split("\n")[:-1][1:]
    )
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values


class BenchmarkEvaluator:
    def __init__(self, split, k=1, chat_format=None):
        self.split = split
        self.k = k
        # initialize all the evaluators
        self.evaluators = {
            "ARC": ARC_Evaluator(split, k, chat_format),
            "BGQA": BGQA_Evaluator(split, k, chat_format),
            "CoinFlip": CoinFlip_Evaluator(split, k, chat_format),
            "ConditionalQA": ConditionalQA_Evaluator(split, k, chat_format),
            "GSM8K": GSM8K_Evaluator(split, k, chat_format),
            "HotpotQA": HotpotQA_Evaluator(split, k, chat_format),
            "LCC": LLC_Evaluator(split, k, chat_format),
            "Quartz": Quartz_Evaluator(split, k, chat_format),
            "StrategyQA": StrategyQA_Evaluator(split, k, chat_format),
        }
        # self.evaluators = {"AQuA": AQuA_Evaluator(split, k, chat_format),
        #                     "CSQA": CSQA_Evaluator(split, k, chat_format),
        #                     "SVAMP": SVAMP_Evaluator(split, k, chat_format)}
        # self.evaluators = {"CSQA": CSQA_Evaluator(split, k, chat_format)}
        # self.evaluators = {"AQuA": AQuA_Evaluator(split, k, chat_format)}
        # self.evaluators = {"klc": kLC_Evaluator(split, k, chat_format)}

    def __call__(
        self,
        llm,
        sampling_params,
        lora_path,
        lora_id=1,
        output_base_path=None,
        postprocess_responses=False,
        hf_inference=False,
        tokenizer=None,
        max_input_length=1024,
        max_new_tokens=4096,
        batch_size=4,
    ):
        main_results = {}
        if output_base_path is None:
            output_base_path = lora_path

        for name, evaluator in self.evaluators.items():
            print(f"Evaluating on {name}")
            st_t = time.time()
            if hf_inference:
                (
                    results,
                    list_preds,
                    list_final_answers,
                    list_responses,
                    list_prompts,
                ) = evaluator.hf_inference(
                    llm, tokenizer, max_input_length, max_new_tokens, batch_size
                )
            else:
                (
                    results,
                    list_preds,
                    list_final_answers,
                    list_responses,
                    list_prompts,
                ) = evaluator(
                    llm, sampling_params, lora_path, lora_id, postprocess_responses
                )
            end_ti = time.time()
            elapsed_min = end_ti - st_t
            print(f"Took {elapsed_min} ms")
            main_metric = evaluator.get_main_metric(results)
            main_results[name] = main_metric

            # save all outputs to a file

            output_path = os.path.join(
                output_base_path,
                self.split,
                f"temp_{sampling_params.temperature}",
                f"evaluation@{self.k}",
                name,
            )
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            with open(os.path.join(output_path, "results.json"), "w") as f:
                json.dump(results, f, indent=4)
            with open(os.path.join(output_path, "list_preds.json"), "w") as f:
                json.dump(list_preds, f, indent=4)
            with open(os.path.join(output_path, "list_final_answers.json"), "w") as f:
                json.dump(list_final_answers, f, indent=4)
            with open(os.path.join(output_path, "list_responses.json"), "w") as f:
                json.dump(list_responses, f, indent=4)
            with open(os.path.join(output_path, "list_prompts.json"), "w") as f:
                json.dump(list_prompts, f, indent=4)

        # create a pandas dataframe for the results
        # the columns are the benchmarks and the row is the main metric
        df = pd.DataFrame(main_results, index=[0])
        with open(
            os.path.join(
                output_base_path,
                self.split,
                f"temp_{sampling_params.temperature}",
                f"evaluation@{self.k}",
                "results.csv",
            ),
            "w",
        ) as f:
            df.to_csv(f, index=False)
        return df

    def self_consistency(
        self,
        llm,
        sampling_params,
        lora_path,
        lora_id=1,
        output_base_path=None,
        postprocess_responses=False,
        self_consistency_k=1,
    ):
        main_results = {}
        if output_base_path is None:
            output_base_path = lora_path

        for name, evaluator in self.evaluators.items():
            print(f"Evaluating on {name}")
            st_t = time.time()
            results, list_preds, list_final_answers, list_responses, list_prompts = (
                evaluator.self_consistency(
                    llm,
                    sampling_params,
                    lora_path=lora_path,
                    lora_id=lora_id,
                    postprocess_responses=postprocess_responses,
                    self_consistency_k=self_consistency_k,
                )
            )
            end_ti = time.time()
            elapsed_min = end_ti - st_t
            print(f"Took {elapsed_min} ms")
            for k, results_k in results.items():
                main_metric = evaluator.get_main_metric(results_k)
                if name not in main_results:
                    main_results[name] = {}
                main_results[name][k] = main_metric

            # save all outputs to a file

            output_path = os.path.join(
                output_base_path,
                self.split,
                f"temp_{sampling_params.temperature}",
                f"self_consistency",
                name,
            )
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            with open(os.path.join(output_path, "results.json"), "w") as f:
                json.dump(results, f, indent=4)
            with open(os.path.join(output_path, "list_preds.json"), "w") as f:
                json.dump(list_preds, f, indent=4)
            with open(os.path.join(output_path, "list_final_answers.json"), "w") as f:
                json.dump(list_final_answers, f, indent=4)
            with open(os.path.join(output_path, "list_responses.json"), "w") as f:
                json.dump(list_responses, f, indent=4)
            with open(os.path.join(output_path, "list_prompts.json"), "w") as f:
                json.dump(list_prompts, f, indent=4)

        # create a pandas dataframe for the results
        list_rows = []
        for dataset_name, results_k in main_results.items():
            list_rows.append([dataset_name] + list(results_k.values()))
            list_columns = ["Dataset"] + list(results_k.keys())
        df = pd.DataFrame(list_rows, columns=list_columns)
        with open(
            os.path.join(
                output_base_path,
                self.split,
                f"temp_{sampling_params.temperature}",
                f"self_consistency",
                f"results.csv",
            ),
            "w",
        ) as f:
            df.to_csv(f, index=False)
        return main_results

    def test_set_eval(
        self,
        dict_task2k,
        chat_format,
        llm,
        sampling_params,
        lora_path,
        lora_id=1,
        output_base_path=None,
        postprocess_responses=False,
        self_consistency=False,
        num_samples_self_consistency=1,
    ):

        if self_consistency:
            return self.dcot_self_consistency(
                dict_task2k,
                chat_format,
                llm,
                sampling_params,
                lora_path,
                lora_id,
                output_base_path,
                postprocess_responses,
                num_samples_self_consistency,
            )

        if output_base_path is None:
            output_base_path = lora_path
        main_results = {}
        evaluators = {
            "ARC": ARC_Evaluator("test", dict_task2k["ARC"], chat_format),
            "BGQA": BGQA_Evaluator("test", dict_task2k["BGQA"], chat_format),
            # "CoinFlip": CoinFlip_Evaluator('test', 2, chat_format),
            "ConditionalQA": ConditionalQA_Evaluator(
                "test", dict_task2k["ConditionalQA"], chat_format
            ),
            "GSM8K": GSM8K_Evaluator("test", dict_task2k["GSM8K"], chat_format),
            "HotpotQA": HotpotQA_Evaluator(
                "test", dict_task2k["HotpotQA"], chat_format
            ),
            "LLC": LLC_Evaluator("test", dict_task2k["LLC"], chat_format),
            "Quartz": Quartz_Evaluator("test", dict_task2k["Quartz"], chat_format),
            "StrategyQA": StrategyQA_Evaluator(
                "test", dict_task2k["StrategyQA"], chat_format
            ),
        }
        for name, evaluator in evaluators.items():
            results, list_preds, list_final_answers, list_responses, list_prompts = (
                evaluator(
                    llm, sampling_params, lora_path, lora_id, postprocess_responses
                )
            )
            main_metric = evaluator.get_main_metric(results)
            main_results[name] = main_metric

            # save all outputs to a file

            output_path = os.path.join(
                output_base_path,
                self.split,
                f"temp_{sampling_params.temperature}",
                name,
            )
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            with open(os.path.join(output_path, "results.json"), "w") as f:
                json.dump(results, f, indent=4)
            with open(os.path.join(output_path, "list_preds.json"), "w") as f:
                json.dump(list_preds, f, indent=4)
            with open(os.path.join(output_path, "list_final_answers.json"), "w") as f:
                json.dump(list_final_answers, f, indent=4)
            with open(os.path.join(output_path, "list_responses.json"), "w") as f:
                json.dump(list_responses, f, indent=4)
            with open(os.path.join(output_path, "list_prompts.json"), "w") as f:
                json.dump(list_prompts, f, indent=4)

        # create a pandas dataframe for the results
        df = pd.DataFrame(main_results, index=[0])
        with open(
            os.path.join(
                output_base_path,
                self.split,
                f"temp_{sampling_params.temperature}",
                f"results.csv",
            ),
            "w",
        ) as f:
            df.to_csv(f, index=False)
        return main_results

    def dcot_self_consistency(
        self,
        dict_task2k,
        chat_format,
        llm,
        sampling_params,
        lora_path,
        lora_id=1,
        output_base_path=None,
        postprocess_responses=False,
        self_consistency_k=1,
    ):
        if output_base_path is None:
            output_base_path = lora_path
        main_results = {}
        evaluators = {
            "ARC": ARC_Evaluator("test", dict_task2k["ARC"], chat_format),
            "BGQA": BGQA_Evaluator("test", dict_task2k["BGQA"], chat_format),
            # "CoinFlip": CoinFlip_Evaluator('test', 2, chat_format),
            "ConditionalQA": ConditionalQA_Evaluator(
                "test", dict_task2k["ConditionalQA"], chat_format
            ),
            "GSM8K": GSM8K_Evaluator("test", dict_task2k["GSM8K"], chat_format),
            "HotpotQA": HotpotQA_Evaluator(
                "test", dict_task2k["HotpotQA"], chat_format
            ),
            "LLC": LLC_Evaluator("test", dict_task2k["LLC"], chat_format),
            "Quartz": Quartz_Evaluator("test", dict_task2k["Quartz"], chat_format),
            "StrategyQA": StrategyQA_Evaluator(
                "test", dict_task2k["StrategyQA"], chat_format
            ),
        }
        for name, evaluator in evaluators.items():
            results, list_preds, list_final_answers, list_responses, list_prompts = (
                evaluator.self_consistency(
                    llm,
                    sampling_params,
                    lora_path=lora_path,
                    lora_id=lora_id,
                    postprocess_responses=postprocess_responses,
                    self_consistency_k=self_consistency_k,
                )
            )

            for k, results_k in results.items():
                main_metric = evaluator.get_main_metric(results_k)
                if name not in main_results:
                    main_results[name] = {}
                main_results[name][k] = main_metric

            # save all outputs to a file

            output_path = os.path.join(
                output_base_path,
                self.split,
                f"temp_{sampling_params.temperature}",
                f"self_consistency",
                name,
            )
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            with open(os.path.join(output_path, "results.json"), "w") as f:
                json.dump(results, f, indent=4)
            with open(os.path.join(output_path, "list_preds.json"), "w") as f:
                json.dump(list_preds, f, indent=4)
            with open(os.path.join(output_path, "list_final_answers.json"), "w") as f:
                json.dump(list_final_answers, f, indent=4)
            with open(os.path.join(output_path, "list_responses.json"), "w") as f:
                json.dump(list_responses, f, indent=4)
            with open(os.path.join(output_path, "list_prompts.json"), "w") as f:
                json.dump(list_prompts, f, indent=4)

        # create a pandas dataframe for the results
        list_rows = []
        for dataset_name, results_k in main_results.items():
            list_rows.append([dataset_name] + list(results_k.values()))
            list_columns = ["Dataset"] + list(results_k.keys())
        df = pd.DataFrame(list_rows, columns=list_columns)
        with open(
            os.path.join(
                output_base_path,
                self.split,
                f"temp_{sampling_params.temperature}",
                f"self_consistency",
                f"results.csv",
            ),
            "w",
        ) as f:
            df.to_csv(f, index=False)
        return main_results


class Evaluator:

    def __init__(self, split):
        self.split = split
        self.prompts = []
        self.cnt_error_response_processing = 0

    def __call__(
        self, llm, sampling_params, lora_path, lora_id=1, postprocess_responses=False
    ):
        lora_request = None
        if lora_path is not None:
            adapter_name = "_".join(lora_path.split("/"))
            lora_request = LoRARequest(adapter_name, lora_id, lora_path)
        outputs = llm.generate(
            self.prompts,
            sampling_params,
            lora_request=lora_request,
        )
        list_responses = []
        list_final_answers = []
        for x in outputs:
            response = x.outputs[0].text
            if postprocess_responses:
                response = self.clean_output(response)
            list_responses.append(response)
            final_answer = self.get_final_answer(response)
            list_final_answers.append(final_answer)

        list_preds = []
        for response in list_final_answers:
            pred = self.process_response(response)
            list_preds.append(pred)

        results = self.evaluate(list_preds)
        return (
            results,
            list_preds,
            list_final_answers,
            list_responses,
            self.prompts.list,
        )

    def hf_inference(
        self,
        llm,
        tokenizer,
        max_input_length=1024,
        max_new_tokens=4096,
        batch_size=4,
    ):

        list_responses = []
        list_final_answers = []
        for i in trange(0, len(self.prompts), batch_size):
            inputs = tokenizer(
                [p for p in self.prompts.list[i : i + batch_size]],
                return_tensors="pt",
                max_length=max_input_length,
                truncation=True,
                padding=True,
            )

            output_sequences = llm.generate(
                input_ids=inputs["input_ids"].to("cuda"),
                attention_mask=inputs["attention_mask"].to("cuda"),
                max_new_tokens=max_new_tokens - max_input_length,
                do_sample=False,
            )

            outputs = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
            for response in outputs:
                list_responses.append(response)
                final_answer = self.get_final_answer(response)
                list_final_answers.append(final_answer)

            print("Free GPU: ", get_gpu_memory())

        list_preds = []
        for response in list_final_answers:
            pred = self.process_response(response)
            list_preds.append(pred)

        results = self.evaluate(list_preds)
        return (
            results,
            list_preds,
            list_final_answers,
            list_responses,
            self.prompts.list,
        )

    def self_consistency(
        self,
        llm,
        sampling_params,
        lora_path,
        lora_id=1,
        postprocess_responses=False,
        self_consistency_k=1,
    ):
        list_preds = []
        list_final_answers = []
        list_responses = []
        for k in range(1, self_consistency_k + 1):
            (_, list_preds_k, list_final_answers_k, list_responses_k, _) = (
                self.__call__(
                    llm, sampling_params, lora_path, lora_id, postprocess_responses
                )
            )

            list_preds = self._append_sample_generations(list_preds, list_preds_k)
            list_final_answers = self._append_sample_generations(
                list_final_answers, list_final_answers_k
            )
            list_responses = self._append_sample_generations(
                list_responses, list_responses_k
            )

        results = dict()
        for k in range(1, self_consistency_k + 1):
            list_preds_k = []
            for preds in list_preds:
                list_preds_k.append(self._most_common(preds[:k]))
            results_k = self.evaluate(list_preds_k)
            results[k] = results_k
        return (
            results,
            list_preds,
            list_final_answers,
            list_responses,
            self.prompts.list,
        )

    def _append_sample_generations(self, list_self_consistency, list_generations_k):
        if len(list_self_consistency) == 0:
            # first sample
            list_self_consistency = [[pred] for pred in list_generations_k]
        else:
            for i, x in enumerate(list_self_consistency):
                x.append(list_generations_k[i])
        return list_self_consistency

    def clean_output(self, text):
        try:
            split_text = re.split(r"\[.*?\]", text.split("[Final answer]")[1])
            return (
                text.split("[Final answer]")[0]
                + "[Final answer] "
                + split_text[0].strip()
            )
        except:
            return text

    def accuracy(self, list_labels, list_predictions):
        return sum(np.array(list_predictions) == np.array(list_labels)) / len(
            list_predictions
        )

    def _most_common(self, lst):
        data = collections.Counter(lst)
        return data.most_common(1)[0][0]

    def evaluate(self, list_preds):
        raise NotImplementedError

    def get_main_metric(self, results):
        raise NotImplementedError

    def process_response(self, response):
        raise NotImplementedError

    def get_final_answer(self, response):
        try:
            return response.split("[Final answer]")[1].lower().strip()
        except:
            self.cnt_error_response_processing += 1
            print("Error extracting final answer ", self.cnt_error_response_processing)
            return response

    def __len__(self):
        return len(self.prompts)


class ClassificationEvaluator(Evaluator):
    def __init__(self, split):
        super().__init__(split)
        self.labels = None  # must be set in the subclass

    def evaluate(self, list_preds):
        results = classification_report(
            self.labels, list_preds, output_dict=True, zero_division=0
        )
        return results

    def get_main_metric(self, results):
        return results["macro avg"]["f1-score"]


class PromptDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, k=1, chat_format=None):
        self.list = []
        for x in dataset:
            prompt = str(
                Prompt(
                    question=x["question"],
                    k=k,
                    options=x["options"],
                    context=x["context"],
                    chat_format=chat_format,
                )
            )
            self.list.append(prompt)

    def __len__(self):
        return len(self.list)

    def __getitem__(self, i):
        return self.list[i]


class AQuA_Evaluator(ClassificationEvaluator):
    def __init__(self, split, k=1, chat_format=None):
        super().__init__(split)

        if split == "validation":
            raw_dataset = []
            with open("data/aqua/dev.json") as f:
                lines = f.readlines()
            for l in lines:
                raw_dataset.append(json.loads(l))
        elif split == "test":
            raw_dataset = []
            with open("data/aqua/test.json") as f:
                lines = f.readlines()
            for l in lines:
                raw_dataset.append(json.loads(l))
        else:
            raise ValueError("Invalid split")

        self.dataset = []
        for x in raw_dataset:
            self.dataset.append(
                {
                    "question": x["question"],
                    "options": x["options"],
                    "context": None,
                }
            )
        self.prompts = PromptDataset(self.dataset, k=k, chat_format=chat_format)
        self.labels = [c["correct"] for c in raw_dataset]

    def process_response(self, response):
        """
        Extracts the final conclusion of the model and returns True or False.
        """
        response_lower = response.lower()
        try:
            if "a)" in response_lower:
                return "A"
            elif "b)" in response_lower:
                return "B"
            elif "c)" in response_lower:
                return "C"
            elif "d)" in response_lower:
                return "D"
            else:
                return "A"
        except:
            return "A"


class ARC_Evaluator(ClassificationEvaluator):
    def __init__(self, split, k=1, chat_format=None):
        super().__init__(split)
        raw_dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge")

        # prepare format
        create_options = lambda x: " ".join(
            [f"{chr(65 + i)}) {opt}" for i, opt in enumerate(x["choices"]["text"])]
        )
        if split == "validation":
            raw_dataset = raw_dataset["validation"]

        elif split == "test":
            raw_dataset = raw_dataset["test"]
        elif split == "70B":
            raw_dataset = raw_dataset["validation"]
            with open("data/arc_hard/llama70b_dev_idx.json") as f:
                idx = json.load(f)
            raw_dataset = raw_dataset.select(idx)

        # remove corrupted data points
        clean_dataset = []
        for i in range(len(raw_dataset)):
            if raw_dataset["answerKey"][i] in ["A", "B", "C", "D"]:
                clean_dataset.append(raw_dataset[i])
        for x in clean_dataset:
            x["options"] = create_options(x)
            x["context"] = None
        self.dataset = clean_dataset
        self.prompts = PromptDataset(self.dataset, k=k, chat_format=chat_format)
        self.labels = [c["answerKey"] for c in self.dataset]

    def process_response(self, response):
        """
        Extracts the final conclusion of the model and returns True or False.
        """
        response_lower = response.lower()
        try:
            if "a)" in response_lower:
                return "A"
            elif "b)" in response_lower:
                return "B"
            elif "c)" in response_lower:
                return "C"
            elif "d)" in response_lower:
                return "D"
            else:
                return "A"
        except:
            return "A"


class BGQA_Evaluator(ClassificationEvaluator):
    def __init__(self, split, k=1, chat_format=None):
        super().__init__(split)
        if split == "validation":
            with open("data/boardgameqa/BoardgameQA-Main-depth3/valid.json", "r") as f:
                raw_dataset = json.load(f)
        elif split == "test":
            with open("data/boardgameqa/BoardgameQA-Main-depth3/test.json", "r") as f:
                raw_dataset = json.load(f)
        elif split == "70B":
            with open("data/boardgameqa/BoardgameQA-Main-depth3/valid.json", "r") as f:
                raw_dataset = json.load(f)
            with open(
                "data/boardgameqa/BoardgameQA-Main-depth3/llama70b_dev_idx.json"
            ) as f:
                idx = json.load(f)
            raw_dataset = [raw_dataset[i] for i in idx]
        else:
            raise ValueError("Invalid split")

        self.dataset = []
        for x in raw_dataset:
            self.dataset.append(
                {
                    "question": x["example"],
                    "options": "A) Yes B) No C) Unknown",
                    "context": None,
                }
            )
        self.prompts = PromptDataset(self.dataset, k=k, chat_format=chat_format)
        self.labels = []
        for x in raw_dataset:
            if x["label"] == "proved":
                self.labels.append("A")
            elif x["label"] == "disproved":
                self.labels.append("B")
            elif x["label"] == "unknown":
                self.labels.append("C")
            else:
                raise ValueError("Invalid label")

    def process_response(self, response):
        if "a)" in response.lower():
            return "A"
        elif "b)" in response.lower():
            return "B"
        elif "c)" in response.lower():
            return "C"
        tokens = nltk.word_tokenize(response.lower())
        if "yes" in tokens:
            return "A"
        elif "no" in tokens:
            return "B"
        else:
            return "C"


class CoinFlip_Evaluator(ClassificationEvaluator):
    def __init__(self, split, k=1, chat_format=None):
        super().__init__(split)
        raw_dataset = load_dataset("skrishna/coin_flip")

        if split == "validation":
            raw_dataset = raw_dataset["validation"]
        elif split == "test":
            raw_dataset = raw_dataset["test"]
        elif split == "70B":
            # we don't use coinflip for 70B but I am keeping this for consistency
            raw_dataset = raw_dataset["validation"]
        else:
            raise ValueError("Invalid split")
        self.dataset = []
        for x in raw_dataset:
            self.dataset.append(
                {
                    "question": x["inputs"],
                    "options": "A) Yes B) No",
                    "context": None,
                }
            )
        self.prompts = PromptDataset(self.dataset, k=k, chat_format=chat_format)
        self.labels = []
        for x in raw_dataset:
            if x["targets"] == "yes":
                self.labels.append("A")
            elif x["targets"] == "no":
                self.labels.append("B")

    def process_response(self, response):
        if "a)" in response.lower():
            return "A"
        elif "b)" in response.lower():
            return "B"
        tokens = nltk.word_tokenize(response.lower())
        if "yes" in tokens:
            return "A"
        else:
            return "B"


class CSQA_Evaluator(ClassificationEvaluator):
    def __init__(self, split, k=1, chat_format=None):
        super().__init__(split)
        raw_dataset = load_dataset("tau/commonsense_qa")

        if split == "validation":
            raw_dataset = raw_dataset["validation"]
        elif split == "test":
            # raw_dataset = raw_dataset["test"]
            raw_dataset = raw_dataset["validation"]
            # test set doesn't have labels
        else:
            raise ValueError("Invalid split")

        create_options = lambda x: " ".join(
            [f"{chr(65 + i)}) {opt}" for i, opt in enumerate(x["choices"]["text"])]
        )

        self.dataset = []
        for x in raw_dataset:
            self.dataset.append(
                {
                    "question": x["question"],
                    "options": create_options(x),
                    "context": None,
                }
            )
        self.prompts = PromptDataset(self.dataset, k=k, chat_format=chat_format)
        self.labels = [c["answerKey"] for c in raw_dataset]

    def process_response(self, response):
        """
        Extracts the final conclusion of the model and returns True or False.
        """
        response_lower = response.lower()
        try:
            if "a)" in response_lower:
                return "A"
            elif "b)" in response_lower:
                return "B"
            elif "c)" in response_lower:
                return "C"
            elif "d)" in response_lower:
                return "D"
            else:
                return "A"
        except:
            return "A"


class ConditionalQA_Evaluator(Evaluator):
    def __init__(self, split, k=1, chat_format=None):
        super().__init__(split)

        if split == "validation":
            with open("data/conditionalqa/dev.json", "r") as f:
                raw_dataset = json.load(f)
        elif split == "test":
            with open("data/conditionalqa/test.json", "r") as f:
                raw_dataset = json.load(f)
        elif split == "70B":
            with open("data/conditionalqa/dev.json", "r") as f:
                raw_dataset = json.load(f)
            with open("data/conditionalqa/llama70b_dev_idx.json", "r") as f:
                idx = json.load(f)
            raw_dataset = [raw_dataset[i] for i in idx]
        else:
            raise ValueError("Invalid split")

        with open("data/conditionalqa/documents.json") as f:
            docs = json.load(f)

        url2doc = {d["url"]: d for i, d in enumerate(docs)}
        self.dataset = []
        for x in raw_dataset:
            self.dataset.append(
                {
                    "question": x["scenario"] + " " + x["question"],
                    "options": None,
                    "context": self.get_summarized_doc(x, url2doc),
                }
            )
        self.prompts = PromptDataset(self.dataset, k=k, chat_format=chat_format)
        self.labels = []
        for x in raw_dataset:
            if x["not_answerable"]:
                self.labels.append([""])
            else:
                self.labels.append([ans_tup[0] for ans_tup in x["answers"]])

    def evaluate(self, list_preds):
        # evaluate using squad metric
        squad_metric = load("squad")
        predictions = [
            {"id": str(i), "prediction_text": p} for i, p in enumerate(list_preds)
        ]
        references = [
            {"id": str(i), "answers": {"answer_start": [0] * len(l), "text": l}}
            for i, l in enumerate(self.labels)
        ]
        results = squad_metric.compute(predictions=predictions, references=references)
        for metric in results.keys():
            results[metric] /= 100
        return results

    def process_response(self, response):
        return response

    def get_main_metric(self, results):
        return results["f1"]

    def get_summarized_doc(self, x, url2doc):
        """
        Oracle retriever for the conditionalQA dataset.
        Returns the contextualized rationales for the given example.
        Contextualized rationales are defined as the sections that contain the rationales.
        """
        doc = url2doc[x["url"]]
        list_sections = self.get_sections(doc["contents"])
        summarized_doc = self.create_contextualized_rationales(
            list_sections, x["evidences"]
        )
        return summarized_doc

    def get_sections(self, doc):
        """
        This function takes in a document as input and returns a list of sections.
        A section is defined as a list of tags that are enclosed by a header tag (h1, h2, h3, or h4).
        """
        list_sections = []
        section = []
        for tag in doc:
            if "<h1>" in tag or "<h2>" in tag or "<h3>" in tag or "<h4>" in tag:
                if len(section) > 0:
                    list_sections.append(section)
                section = []

            section.append(tag)
        if len(section) > 0:
            list_sections.append(section)
        return list_sections

    def create_contextualized_rationales(self, list_sections, list_rationales):
        """
        This function takes in two lists: list_sections and list_rationales.
        It returns a string that contains the contextualized rationales.
        The function first adds the first section of list_sections to the output list.
        Then, for each section in list_sections, it checks if any of the rationales in list_rationales
        are present in the section. If so, it adds the section to the output list and moves on to the next section.
        The output list is then flattened and joined by newline characters to create the final output string.
        """
        contextualized_rationales = []
        # always add the first section, which is usually an overview
        contextualized_rationales.append(list_sections[0])
        for section in list_sections[1:]:
            for rationale in list_rationales:
                if rationale in section:
                    contextualized_rationales.append(section)
                    break
        # flatten the list
        contextualized_rationales = [
            item for sublist in contextualized_rationales for item in sublist
        ]
        # join by \n
        contextualized_rationales = "\n".join(contextualized_rationales)
        return contextualized_rationales


class GSM8K_Evaluator(Evaluator):
    def __init__(self, split, k=1, chat_format=None):
        super().__init__(split)
        raw_dataset = load_dataset("gsm8k", "main")
        if split == "validation":
            with open("data/gsm8k/validation_ids.json") as f:
                validation_ids = json.load(f)
            raw_dataset = [
                x for i, x in enumerate(raw_dataset["train"]) if i in validation_ids
            ]
        elif split == "test":
            raw_dataset = raw_dataset["test"]
        elif split == "70B":
            with open("data/gsm8k/validation_ids.json") as f:
                validation_ids = json.load(f)
            raw_dataset = [
                x for i, x in enumerate(raw_dataset["train"]) if i in validation_ids
            ]
            with open("data/gsm8k/llama70b_dev_idx.json") as f:
                idx = json.load(f)
            raw_dataset = [raw_dataset[i] for i in idx]
        else:
            raise ValueError("Invalid split")
        self.dataset = []
        for x in raw_dataset:
            self.dataset.append(
                {
                    "question": x["question"],
                    "options": None,
                    "context": None,
                }
            )
        self.prompts = PromptDataset(self.dataset, k=k, chat_format=chat_format)
        self.labels = []
        for x in raw_dataset:
            self.labels.append(x["answer"].split("####")[1].strip())

    def evaluate(self, list_preds):
        return {"accuracy": self.accuracy(self.labels, list_preds)}

    def get_main_metric(self, results):
        return results["accuracy"]

    def process_response(self, response):
        return response


class HotpotQA_Evaluator(Evaluator):
    def __init__(self, split, k=1, chat_format=None):
        super().__init__(split)

        if split == "validation":
            with open("data/hotpotqa/hotpot_train_v1.1.json", "r") as f:
                raw_dataset = json.load(f)
            with open("data/hotpotqa/validation_ids.json", "r") as f:
                validation_ids = json.load(f)
            raw_dataset = [x for i, x in enumerate(raw_dataset) if i in validation_ids]
        elif split == "test":
            with open("data/hotpotqa/hotpot_dev_distractor_v1.json", "r") as f:
                raw_dataset = json.load(f)
        elif split == "70B":
            with open("data/hotpotqa/hotpot_train_v1.1.json", "r") as f:
                raw_dataset = json.load(f)
            with open("data/hotpotqa/validation_ids.json", "r") as f:
                validation_ids = json.load(f)
            raw_dataset = [x for i, x in enumerate(raw_dataset) if i in validation_ids]
            with open("data/hotpotqa/llama70b_dev_idx.json", "r") as f:
                idx = json.load(f)
            raw_dataset = [raw_dataset[i] for i in idx]
        else:
            raise ValueError("Invalid split")
        self.dataset = []
        for x in raw_dataset:
            self.dataset.append(
                {
                    "question": x["question"],
                    "context": self.get_full_context(x),
                    "options": None,
                }
            )
        self.prompts = PromptDataset(self.dataset, k=k, chat_format=chat_format)
        self.labels = []
        for x in raw_dataset:
            self.labels.append(x["answer"])

    def evaluate(self, list_preds):
        # evaluate using squad metric
        squad_metric = load("squad")
        predictions = [
            {"id": str(i), "prediction_text": p} for i, p in enumerate(list_preds)
        ]
        references = [
            {"id": str(i), "answers": {"answer_start": [0], "text": [l]}}
            for i, l in enumerate(self.labels)
        ]
        results = squad_metric.compute(predictions=predictions, references=references)
        for metric in results.keys():
            results[metric] /= 100
        return results

    def get_main_metric(self, results):
        return results["f1"]

    def process_response(self, response):
        return response

    def get_full_context(self, x):
        context = ""
        for doc in x["context"]:
            context += doc[0] + "\n"
            context += ". ".join(doc[1]) + "\n\n"
        context = context.strip()
        return context


class LLC_Evaluator(Evaluator):
    def __init__(self, split, k=1, chat_format=None):
        super().__init__(split)

        raw_dataset = load_dataset("ChilleD/LastLetterConcat")
        if split == "validation":
            raw_dataset = raw_dataset["test"].select(range(50))
        elif split == "test":
            raw_dataset = raw_dataset["test"].select(
                range(50, len(raw_dataset["test"]))
            )
        elif split == "70B":
            raw_dataset = raw_dataset["test"].select(range(50))
        else:
            raise ValueError("Invalid split")
        self.dataset = []
        for x in raw_dataset:
            self.dataset.append(
                {
                    "question": x["question"],
                    "options": None,
                    "context": None,
                }
            )

        self.prompts = PromptDataset(self.dataset, k=k, chat_format=chat_format)
        self.labels = []
        for x in raw_dataset:
            self.labels.append(x["answer"])

    def evaluate(self, list_preds):
        return {"accuracy": self.accuracy(self.labels, list_preds)}

    def get_main_metric(self, results):
        return results["accuracy"]

    def process_response(self, response):
        if "\n" in response:
            clean_pred = response.split("\n")[0].strip()
            if clean_pred == "":
                return response
            return clean_pred
        return response


class Quartz_Evaluator(ClassificationEvaluator):
    def __init__(self, split, k=1, chat_format=None):
        super().__init__(split)
        raw_dataset = load_dataset("allenai/quartz")
        if split == "validation":
            raw_dataset = raw_dataset["validation"]
        elif split == "test":
            raw_dataset = raw_dataset["test"]
        elif split == "70B":
            raw_dataset = raw_dataset["validation"]
            with open("data/quartz/llama70b_dev_idx.json") as f:
                idx = json.load(f)
            raw_dataset = raw_dataset.select(idx)
        else:
            raise ValueError("Invalid split")
        self.dataset = []
        for x in raw_dataset:
            options = " ".join(
                [f"{chr(65 + i)}) {opt}" for i, opt in enumerate(x["choices"]["text"])]
            )
            self.dataset.append(
                {
                    "question": x["question"],
                    "options": options,
                    "context": x["para"],
                }
            )

        self.prompts = PromptDataset(self.dataset, k=k, chat_format=chat_format)
        self.labels = raw_dataset["answerKey"]

    def process_response(self, response):
        if "a)" in response.lower():
            return "A"
        elif "b)" in response.lower():
            return "B"
        return "A"


class StrategyQA_Evaluator(ClassificationEvaluator):
    def __init__(self, split, k=1, chat_format=None):
        super().__init__(split)

        raw_dataset = load_dataset("ChilleD/StrategyQA")
        raw_dataset = raw_dataset["test"]
        if split == "validation":
            with open("data/strategyqa/strategyqa_dev_idx.json") as f:
                idx = json.load(f)
            raw_dataset = raw_dataset.select(idx)
        elif split == "test":
            with open("data/strategyqa/strategyqa_test_idx.json") as f:
                idx = json.load(f)
            raw_dataset = raw_dataset.select(idx)
        elif split == "70B":
            with open("data/strategyqa/llama70b_dev_idx.json") as f:
                idx = json.load(f)
            raw_dataset = raw_dataset.select(idx)
        self.dataset = []
        for x in raw_dataset:
            self.dataset.append(
                {
                    "question": x["question"],
                    "options": "A) Yes B) No",
                    "context": None,
                    "answer": x["answer"],
                }
            )

        self.prompts = PromptDataset(self.dataset, k=k, chat_format=chat_format)
        self.labels = []
        for x in raw_dataset:
            self.labels.append("A" if x["answer"] else "B")

    def process_response(self, response):
        if "a)" in response.lower():
            return "A"
        elif "b)" in response.lower():
            return "B"
        tokens = nltk.word_tokenize(response.lower())
        if "yes" in tokens:
            return "A"
        else:
            return "B"


class SVAMP_Evaluator(Evaluator):
    def __init__(self, split, k=1, chat_format=None):
        super().__init__(split)
        if split == "validation":
            with open("data/svamp/dev.json") as f:
                raw_dataset = json.load(f)
        elif split == "test":
            with open("data/svamp/test.json") as f:
                raw_dataset = json.load(f)
        else:
            raise ValueError("Invalid split")
        self.dataset = []
        for x in raw_dataset:
            self.dataset.append(
                {
                    "question": x["Question"],
                    "options": None,
                    "context": x["Body"],
                }
            )
        self.prompts = PromptDataset(self.dataset, k=k, chat_format=chat_format)
        self.labels = []
        for x in raw_dataset:
            self.labels.append(x["Answer"])

    def evaluate(self, list_preds):
        int_preds = []
        for p in list_preds:
            try:
                int_preds.append(int(p))
            except:
                int_preds.append(0)
        return {"accuracy": self.accuracy(self.labels, int_preds)}

    def get_main_metric(self, results):
        return results["accuracy"]

    def process_response(self, response):
        return response


def evaluate_consistency(responses):
    response_counts = {
        response: responses.count(response) / len(responses) * 100
        for response in set(responses)
    }
    return max(response_counts.values())
