import json
import os
import re
from typing import Any

import pandas as pd
from datasets import load_dataset
from sklearn.metrics import classification_report

from vllm.lora.request import LoRARequest

from .data_processors import Prompt


class BBHEvaluation:
    def __init__(self, k=1):
        self.k = k
        self.bbh_tasks = {
            "BooleanExpressions": BooleanExpressions(k),
            "CausalJudgement": CausalJudgement(k),
            "DateUnderstanding": DateUnderstanding(k),
            "DisambiguationQA": DisambiguationQA(k),
            "DyckLanguages": DyckLanguages(k),
            "FormalFallacies": FormalFallacies(k),
            "GeometricShapes": GeometricShapes(k),
            "Hyperbaton": Hyperbaton(k),
            "LogicalDeduction5Obj": LogicalDeduction5Obj(k),
            "LogicalDeduction7Obj": LogicalDeduction7Obj(k),
            "LogicalDeduction3Obj": LogicalDeduction3Obj(k),
            "MovieRecommendation": MovieRecommendation(k),
            "MultistepArithmeticTwo": MultistepArithmeticTwo(k),
            "Navigate": Navigate(k),
            "ObjectCounting": ObjectCounting(k),
            "PenguinsInATable": PenguinsInATable(k),
            "ReasoningAboutColoredObjects": ReasoningAboutColoredObjects(k),
            "RuinNames": RuinNames(k),
            "SalientTranslationErrorDetection": SalientTranslationErrorDetection(k),
            "Snarks": Snarks(k),
            "SportsUnderstanding": SportsUnderstanding(k),
            "TemporalSequences": TemporalSequences(k),
            "TrackingShuffledObjectsFiveObjects": TrackingShuffledObjectsFiveObjects(k),
            "TrackingShuffledObjectsSevenObjects": TrackingShuffledObjectsSevenObjects(
                k
            ),
            "TrackingShuffledObjectsThreeObjects": TrackingShuffledObjectsThreeObjects(
                k
            ),
            "WebOfLies": WebOfLies(k),
            "WordSorting": WordSorting(k),
        }

    def __call__(
        self,
        llm,
        sampling_params,
        lora_path,
        lora_id=1,
        output_base_path=None,
        postprocess_responses=False,
    ):
        main_results = {}
        if output_base_path is None:
            output_base_path = lora_path

        for name, evaluator in self.bbh_tasks.items():
            print(f"Evaluating on {name}")
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
                "BBH",
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
                "BBH",
                f"temp_{sampling_params.temperature}",
                f"evaluation@{self.k}",
                "results.csv",
            ),
            "w",
        ) as f:
            df.to_csv(f, index=False)
        return df


class BBHTask:
    def __init__(self):
        self.list_prompts = []
        self.labels = []
        self._get_letter_option = lambda i: chr(65 + i)

    def __call__(
        self, llm, sampling_params, lora_path, lora_id=1, postprocess_responses=False
    ):
        lora_request = None
        if lora_path is not None:
            adapter_name = "_".join(lora_path.split("/"))
            lora_request = LoRARequest(adapter_name, lora_id, lora_path)
        outputs = llm.generate(
            [str(p) for p in self.list_prompts],
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
            [str(p) for p in self.list_prompts],
        )

    def evaluate(self, list_preds):
        results = classification_report(
            self.labels, list_preds, output_dict=True, zero_division=0
        )
        return results

    def get_main_metric(self, results):
        return results["macro avg"]["f1-score"]

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

    def process_response(self, response):
        response_lower = response.lower()
        label_space = list(set(self.labels))
        label_space.sort()
        try:
            for lbl in label_space:
                if lbl.lower() + ")" in response_lower:
                    return lbl
            return "A"
        except:
            return "A"

    def get_final_answer(self, response):
        try:
            return response.split("[Final answer]")[1].lower().strip()
        except:
            print("Error extracting final answer ")
            return response


class BooleanExpressions(BBHTask):
    def __init__(self, k=1):
        super().__init__()
        dataset = load_dataset("maveriq/bigbenchhard", "boolean_expressions")
        for q in dataset["train"]["input"]:
            options = "A) True, B) False"
            p = Prompt(question=q, k=k, options=options)

            self.list_prompts.append(p)

        for lbl in dataset["train"]["target"]:
            if lbl == "True":
                lbl = "A"
            elif lbl == "False":
                lbl = "B"
            else:
                raise ValueError("Invalid label")
            self.labels.append(lbl)


class CausalJudgement(BBHTask):
    def __init__(self, k=1):
        super().__init__()
        dataset = load_dataset("maveriq/bigbenchhard", "causal_judgement")
        for i in dataset["train"]["input"]:
            parts = i.split("\n")
            q = parts[0]
            context = parts[1]
            options = "A) Yes, B) No"
            p = Prompt(question=q, k=k, options=options, context=context)
            self.list_prompts.append(p)
        for lbl in dataset["train"]["target"]:
            if lbl == "Yes":
                lbl = "A"
            elif lbl == "No":
                lbl = "B"
            else:
                raise ValueError("Invalid label")
            self.labels.append(lbl)


class DateUnderstanding(BBHTask):
    def __init__(self, k=1):
        super().__init__()
        dataset = load_dataset("maveriq/bigbenchhard", "date_understanding")
        for i in dataset["train"]["input"]:
            parts = i.split("\n")
            q = parts[0]
            options = [
                f"{self._get_letter_option(i)}){o[3:]}" for i, o in enumerate(parts[2:])
            ]
            options = " ".join(options)
            p = Prompt(question=q, k=k, options=options)
            self.list_prompts.append(p)

        for lbl in dataset["train"]["target"]:
            if "A" in lbl:
                lbl = "A"
            elif "B" in lbl:
                lbl = "B"
            elif "C" in lbl:
                lbl = "C"
            elif "D" in lbl:
                lbl = "D"
            elif "E" in lbl:
                lbl = "E"
            elif "F" in lbl:
                lbl = "F"
            else:
                raise ValueError("Invalid label")

            self.labels.append(lbl)


class DisambiguationQA(BBHTask):
    def __init__(self, k=1):
        super().__init__()
        dataset = load_dataset("maveriq/bigbenchhard", "disambiguation_qa")
        for i in dataset["train"]["input"]:
            parts = i.split("\n")
            q = parts[0]
            context = parts[1]
            options = [
                f"{self._get_letter_option(i)}){o[3:]}" for i, o in enumerate(parts[3:])
            ]
            options = " ".join(options)
            p = Prompt(question=q, k=k, options=options, context=context)
            self.list_prompts.append(p)

        for lbl in dataset["train"]["target"]:
            if "A" in lbl:
                lbl = "A"
            elif "B" in lbl:
                lbl = "B"
            elif "C" in lbl:
                lbl = "C"
            else:
                raise ValueError("Invalid label")

            self.labels.append(lbl)


class DyckLanguages(BBHTask):
    def __init__(self, k=1):
        super().__init__()
        dataset = load_dataset("maveriq/bigbenchhard", "dyck_languages")
        for q in dataset["train"]["input"]:
            p = Prompt(question=q, k=k)
            self.list_prompts.append(p)

        self.labels = [lbl for lbl in dataset["train"]["target"]]

    def process_response(self, response):
        return response

    def evaluate(self, list_preds):
        cnt = 0
        for pred, lbl in zip(list_preds, self.labels):
            if pred.lower() == lbl.lower():
                cnt += 1
        return {"accuracy": cnt / len(list_preds)}

    def get_main_metric(self, results):
        return results["accuracy"]


class FormalFallacies(BBHTask):
    def __init__(self, k=1):
        super().__init__()
        dataset = load_dataset("maveriq/bigbenchhard", "formal_fallacies")
        for i in dataset["train"]["input"]:
            parts = i.split("\n")
            q = parts[1]
            context = parts[0]
            options = [
                f"{self._get_letter_option(i)}){o[1:]}" for i, o in enumerate(parts[3:])
            ]
            options = " ".join(options)
            p = Prompt(question=q, k=k, options=options, context=context)
            self.list_prompts.append(p)

        for lbl in dataset["train"]["target"]:
            if "valid" in lbl:
                lbl = "A"
            elif "invalid" in lbl:
                lbl = "B"
            else:
                raise ValueError("Invalid label")
            self.labels.append(lbl)


class GeometricShapes(BBHTask):
    def __init__(self, k=1):
        super().__init__()
        dataset = load_dataset("maveriq/bigbenchhard", "geometric_shapes")
        for i in dataset["train"]["input"]:
            parts = i.split("\n")
            q = parts[0]
            options = [
                f"{self._get_letter_option(i)}){o[3:]}" for i, o in enumerate(parts[2:])
            ]
            options = " ".join(options)
            p = Prompt(question=q, k=k, options=options)
            self.list_prompts.append(p)
        for lbl in dataset["train"]["target"]:
            lbl = lbl.replace("(", "").replace(")", "")
            self.labels.append(lbl)


class Hyperbaton(BBHTask):
    def __init__(self, k=1):
        super().__init__()
        dataset = load_dataset("maveriq/bigbenchhard", "hyperbaton")
        for i in dataset["train"]["input"]:
            parts = i.split("\n")
            q = parts[0]
            options = [
                f"{self._get_letter_option(i)}){o[3:]}" for i, o in enumerate(parts[2:])
            ]
            options = " ".join(options)
            p = Prompt(question=q, k=k, options=options)
            self.list_prompts.append(p)

        for lbl in dataset["train"]["target"]:
            lbl = lbl.replace("(", "").replace(")", "")
            self.labels.append(lbl)


class LogicalDeduction5Obj(BBHTask):
    def __init__(self, k=1):
        super().__init__()
        dataset = load_dataset("maveriq/bigbenchhard", "logical_deduction_five_objects")
        for i in dataset["train"]["input"]:
            parts = i.split("\n")
            q = parts[0]
            options = [
                f"{self._get_letter_option(i)}){o[3:]}" for i, o in enumerate(parts[2:])
            ]
            options = " ".join(options)
            p = Prompt(question=q, k=k, options=options)
            self.list_prompts.append(p)

        for lbl in dataset["train"]["target"]:
            lbl = lbl.replace("(", "").replace(")", "")
            self.labels.append(lbl)


class LogicalDeduction7Obj(BBHTask):
    def __init__(self, k=1):
        super().__init__()
        dataset = load_dataset(
            "maveriq/bigbenchhard", "logical_deduction_seven_objects"
        )
        for i in dataset["train"]["input"]:
            parts = i.split("\n")
            q = parts[0]
            options = [
                f"{self._get_letter_option(i)}){o[3:]}" for i, o in enumerate(parts[2:])
            ]
            options = " ".join(options)
            p = Prompt(question=q, k=k, options=options)
            self.list_prompts.append(p)

        for lbl in dataset["train"]["target"]:
            lbl = lbl.replace("(", "").replace(")", "")
            self.labels.append(lbl)


class LogicalDeduction3Obj(BBHTask):
    def __init__(self, k=1):
        super().__init__()
        dataset = load_dataset(
            "maveriq/bigbenchhard", "logical_deduction_three_objects"
        )
        for i in dataset["train"]["input"]:
            parts = i.split("\n")
            q = parts[0]
            options = [
                f"{self._get_letter_option(i)}){o[3:]}" for i, o in enumerate(parts[2:])
            ]
            options = " ".join(options)
            p = Prompt(question=q, k=k, options=options)
            self.list_prompts.append(p)

        for lbl in dataset["train"]["target"]:
            lbl = lbl.replace("(", "").replace(")", "")
            self.labels.append(lbl)


class MovieRecommendation(BBHTask):
    def __init__(self, k=1):
        super().__init__()
        dataset = load_dataset("maveriq/bigbenchhard", "movie_recommendation")
        for i in dataset["train"]["input"]:
            parts = i.split("\n")
            q = parts[0]
            options = [
                f"{self._get_letter_option(i)}){o[3:]}" for i, o in enumerate(parts[2:])
            ]
            options = " ".join(options)
            p = Prompt(question=q, k=k, options=options)
            self.list_prompts.append(p)

        for lbl in dataset["train"]["target"]:
            lbl = lbl.replace("(", "").replace(")", "")
            self.labels.append(lbl)


class MultistepArithmeticTwo(BBHTask):
    def __init__(self, k=1):
        super().__init__()
        dataset = load_dataset("maveriq/bigbenchhard", "multistep_arithmetic_two")
        for q in dataset["train"]["input"]:
            p = Prompt(question=q, k=k)
            self.list_prompts.append(p)

        self.labels = [lbl for lbl in dataset["train"]["target"]]

    def process_response(self, response):
        return response

    def evaluate(self, list_preds):
        cnt = 0
        for pred, lbl in zip(list_preds, self.labels):
            if pred.lower() == lbl.lower():
                cnt += 1
        return {"accuracy": cnt / len(list_preds)}

    def get_main_metric(self, results):
        return results["accuracy"]


class Navigate(BBHTask):
    def __init__(self, k=1):
        super().__init__()
        dataset = load_dataset("maveriq/bigbenchhard", "navigate")
        for i in dataset["train"]["input"]:
            parts = i.split("\n")
            q = parts[0]
            options = "A) Yes B) No"
            p = Prompt(question=q, k=k, options=options)
            self.list_prompts.append(p)

        for lbl in dataset["train"]["target"]:
            if lbl == "Yes":
                lbl = "A"
            elif lbl == "No":
                lbl = "B"
            else:
                raise ValueError("Invalid label")
            self.labels.append(lbl)


class ObjectCounting(BBHTask):
    def __init__(self, k=1):
        super().__init__()
        dataset = load_dataset("maveriq/bigbenchhard", "object_counting")
        for q in dataset["train"]["input"]:
            p = Prompt(
                question=q,
                k=k,
            )
            self.list_prompts.append(p)

        self.labels = [lbl for lbl in dataset["train"]["target"]]

    def process_response(self, response):
        return response

    def evaluate(self, list_preds):
        cnt = 0
        for pred, lbl in zip(list_preds, self.labels):
            if pred.lower() == lbl.lower():
                cnt += 1
        return {"accuracy": cnt / len(list_preds)}

    def get_main_metric(self, results):
        return results["accuracy"]


class PenguinsInATable(BBHTask):
    def __init__(self, k=1):
        super().__init__()
        dataset = load_dataset("maveriq/bigbenchhard", "penguins_in_a_table")
        for i in dataset["train"]["input"]:
            parts = i.split("\n")
            q_idx = [s_idx for s_idx, s in enumerate(parts) if "?" in s][0]
            q = parts[q_idx]
            context = " ".join(parts[:q_idx])
            options = i.split("Options:\n")[1]
            options = options.replace("(", "")
            options = options.replace("\n", " ")
            p = Prompt(question=q, k=k, options=options, context=context)
            self.list_prompts.append(p)

        for lbl in dataset["train"]["target"]:
            lbl = lbl.replace("(", "").replace(")", "")
            self.labels.append(lbl)


class ReasoningAboutColoredObjects(BBHTask):
    def __init__(self, k=1):
        super().__init__()
        dataset = load_dataset(
            "maveriq/bigbenchhard", "reasoning_about_colored_objects"
        )
        for i in dataset["train"]["input"]:
            parts = i.split("\n")
            q = parts[0]
            context = parts[1]
            options = [
                f"{self._get_letter_option(i)}){o[3:]}" for i, o in enumerate(parts[3:])
            ]
            options = " ".join(options)
            p = Prompt(question=q, k=k, options=options, context=context)
            self.list_prompts.append(p)

        for lbl in dataset["train"]["target"]:
            lbl = lbl.replace("(", "").replace(")", "")
            self.labels.append(lbl)


class RuinNames(BBHTask):
    def __init__(self, k=1):
        super().__init__()
        dataset = load_dataset("maveriq/bigbenchhard", "ruin_names")
        for i in dataset["train"]["input"]:
            parts = i.split("\n")
            q = parts[0]
            options = [
                f"{self._get_letter_option(i)}){o[3:]}" for i, o in enumerate(parts[2:])
            ]
            options = " ".join(options)
            p = Prompt(question=q, k=k, options=options)
            self.list_prompts.append(p)

        for lbl in dataset["train"]["target"]:
            lbl = lbl.replace("(", "").replace(")", "")
            self.labels.append(lbl)


class SalientTranslationErrorDetection(BBHTask):
    def __init__(self, k=1):
        super().__init__()
        dataset = load_dataset(
            "maveriq/bigbenchhard", "salient_translation_error_detection"
        )
        for i in dataset["train"]["input"]:
            parts = i.split("\n")
            q = parts[2]
            context = parts[0] + " " + parts[1]
            options = [
                f"{self._get_letter_option(i)}){o[3:]}" for i, o in enumerate(parts[4:])
            ]
            options = " ".join(options)
            p = Prompt(question=q, k=k, options=options, context=context)
            self.list_prompts.append(p)

        for lbl in dataset["train"]["target"]:
            lbl = lbl.replace("(", "").replace(")", "")
            self.labels.append(lbl)


class Snarks(BBHTask):
    def __init__(self, k=1):
        super().__init__()
        dataset = load_dataset("maveriq/bigbenchhard", "snarks")
        for i in dataset["train"]["input"]:
            parts = i.split("\n")
            q = parts[0]
            options = [
                f"{self._get_letter_option(i)}){o[3:]}" for i, o in enumerate(parts[2:])
            ]
            options = " ".join(options)
            p = Prompt(question=q, k=k, options=options)
            self.list_prompts.append(p)

        for lbl in dataset["train"]["target"]:
            lbl = lbl.replace("(", "").replace(")", "")
            self.labels.append(lbl)


class SportsUnderstanding(BBHTask):
    def __init__(self, k=1):
        super().__init__()
        dataset = load_dataset("maveriq/bigbenchhard", "sports_understanding")
        for q in dataset["train"]["input"]:
            options = "A) Yes, B) No"
            p = Prompt(
                question=q,
                k=k,
                options=options,
            )
            self.list_prompts.append(p)

        for lbl in dataset["train"]["target"]:
            if lbl == "yes":
                lbl = "A"
            elif lbl == "no":
                lbl = "B"
            else:
                raise ValueError("Invalid label")
            self.labels.append(lbl)


class TemporalSequences(BBHTask):
    def __init__(self, k=1):
        super().__init__()
        dataset = load_dataset("maveriq/bigbenchhard", "temporal_sequences")
        for i in dataset["train"]["input"]:
            parts = i.split("\n")
            q = parts[0]
            context = " ".join(parts[1:9])
            options = i.split("Options:\n")[1]
            options = options.replace("(", "")
            options = options.replace("\n", " ")
            p = Prompt(question=q, k=k, options=options, context=context)
            self.list_prompts.append(p)

        for lbl in dataset["train"]["target"]:
            lbl = lbl.replace("(", "").replace(")", "")
            self.labels.append(lbl)


class TrackingShuffledObjectsFiveObjects(BBHTask):
    def __init__(self, k=1):
        super().__init__()
        dataset = load_dataset(
            "maveriq/bigbenchhard", "tracking_shuffled_objects_five_objects"
        )
        for i in dataset["train"]["input"]:
            parts = i.split("\n")
            q = " ".join(parts[0:2])
            options = [
                f"{self._get_letter_option(i)}){o[3:]}" for i, o in enumerate(parts[3:])
            ]
            options = " ".join(options)
            p = Prompt(question=q, k=k, options=options)
            self.list_prompts.append(p)

        for lbl in dataset["train"]["target"]:
            lbl = lbl.replace("(", "").replace(")", "")
            self.labels.append(lbl)


class TrackingShuffledObjectsSevenObjects(BBHTask):
    def __init__(self, k=1):
        super().__init__()
        dataset = load_dataset(
            "maveriq/bigbenchhard", "tracking_shuffled_objects_seven_objects"
        )
        for i in dataset["train"]["input"]:
            parts = i.split("\n")
            q = " ".join(parts[0:2])
            options = [
                f"{self._get_letter_option(i)}){o[3:]}" for i, o in enumerate(parts[3:])
            ]
            options = " ".join(options)
            p = Prompt(question=q, k=k, options=options)
            self.list_prompts.append(p)

        for lbl in dataset["train"]["target"]:
            lbl = lbl.replace("(", "").replace(")", "")
            self.labels.append(lbl)


class TrackingShuffledObjectsThreeObjects(BBHTask):
    def __init__(self, k=1):
        super().__init__()
        dataset = load_dataset(
            "maveriq/bigbenchhard", "tracking_shuffled_objects_three_objects"
        )
        for i in dataset["train"]["input"]:
            parts = i.split("\n")
            q = " ".join(parts[0:2])
            options = [
                f"{self._get_letter_option(i)}){o[3:]}" for i, o in enumerate(parts[3:])
            ]
            options = " ".join(options)
            p = Prompt(question=q, k=k, options=options)
            self.list_prompts.append(p)

        for lbl in dataset["train"]["target"]:
            lbl = lbl.replace("(", "").replace(")", "")
            self.labels.append(lbl)


class WebOfLies(BBHTask):
    def __init__(self, k=1):
        super().__init__()
        dataset = load_dataset("maveriq/bigbenchhard", "web_of_lies")
        for i in dataset["train"]["input"]:
            parts = i.split("\n")
            q = parts[0]
            options = "A) Yes, B) No"
            p = Prompt(question=q, k=k, options=options)
            self.list_prompts.append(p)

        for lbl in dataset["train"]["target"]:
            if lbl == "Yes":
                lbl = "A"
            elif lbl == "No":
                lbl = "B"
            else:
                raise ValueError("Invalid label")
            self.labels.append(lbl)


class WordSorting(BBHTask):
    def __init__(self, k=1):
        super().__init__()
        dataset = load_dataset("maveriq/bigbenchhard", "word_sorting")
        for q in dataset["train"]["input"]:
            p = Prompt(question=q, k=k)
            self.list_prompts.append(p)

        self.labels = [lbl for lbl in dataset["train"]["target"]]

    def process_response(self, response):
        return response

    def evaluate(self, list_preds):
        cnt = 0
        for pred, lbl in zip(list_preds, self.labels):
            if pred.lower() == lbl.lower():
                cnt += 1
        return {"accuracy": cnt / len(list_preds)}

    def get_main_metric(self, results):
        return results["accuracy"]
