from collections import defaultdict, Counter
from transformers import AutoConfig
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random
import copy
import json
import csv
import os
import re
import editdistance

from src.probe_llm import load_datasets

plt.rcParams["font.family"] = "Times New Roman"

# TODO: add chatgpt
MODEL_COLORS = {'text-<engine>-001': np.array([0.10588235, 0.61960784, 0.46666667, 1.        ]),
                'Cohere-command': np.array([0.10588235, 0.46666667, 0.61960784, 1.        ]),
                'GPT-3': np.array([0.61960784, 0.10588235, 0.46666667, 1.        ]),
                'text-davinci-002': np.array([0.85098039, 0.37254902, 0.00784314, 1.        ]),
                'text-davinci-003': np.array([0.37254902, 0.00784314, 0.85098039, 1.        ]),
                'ChatGPT': np.array([0.85098039, 0.00784314, 0.37254902, 1.        ]),
                'GPT-4': np.array([0.85098039, 0.00784314, 0.37254902, 1.        ]),
                'Cohere': np.array( [0.45882353, 0.43921569, 0.70196078, 1.        ]),
                'OPT': np.array([0.90588235, 0.16078431, 0.54117647, 1.        ]),
                'EleutherAI': np.array([0.4       , 0.65098039, 0.11764706, 1.        ]),
                'BLOOM': np.array([0.90196078, 0.67058824, 0.00784314, 1.        ]),
                "BlenderBot": np.array([0.4       , 0.4       , 0.4       , 1.        ]),
                "T0": np.array([0.9, 0.1, 0., 1.        ]),
                "Flan-T5": np.array([0., 0., 1., 1.        ])}
PROMPT_GROUPING = {
    "prompt_template_1": "structured",
    "prompt_template_2": "natural",
    "prompt_template_3": "structured",
    "prompt_template_4": "structured",
    "prompt_template_5": "natural",
    "prompt_template_6": "natural",
}
LINESTYLE_GROUPING = {
    'text-<engine>-001': "dashdot",
    'Cohere-command': "dashdot",
    'GPT-3': "solid",
    'text-davinci-002': "dashdot",
    'text-davinci-003': "dashdot",
    'GPT-4': "dashdot",
    'ChatGPT': "dashdot",
    'Cohere': "solid",
    'OPT': "solid",
    'EleutherAI': "solid",
    'BLOOM': "solid",
    "BlenderBot": "dotted",
    "T0": "dashed",
    "Flan-T5": "dashed"
}
GROUP_LINESTYLE = {"structured": "dashed", "natural": "dotted"}
HUMAN_AVG_PERFORMANCE = 86.23333333333333
HUMAN_BEST_PERFORMANCE = 89.83333333333333
NUM_PARAMETERS = {
    "openai": {
        "text-ada-001": {
            "parameters": 350 * 10**6,
            "d_model": 1024,
            "num_layers": 24,
            "d_attn": 16,
            "d_ff": 0,
        },
        "text-babbage-001": {
            "parameters": 1.3 * 10**9,
            "d_model": 2048,
            "num_layers": 24,
            "d_attn": 24,
            "d_ff": 0,
        },
        "text-curie-001": {
            "parameters": 6.7 * 10**9,
            "d_model": 4096,
            "num_layers": 32,
            "d_attn": 32,
            "d_ff": 0,
        },
        "text-davinci-001": {
            "parameters": 175 * 10**9,
            "d_model": 12288,
            "num_layers": 96,
            "d_attn": 96,
            "d_ff": 0,
        },
        "objective": "LM",
        "training data size": 300 * 10**9,
        "compute": 0,
        "model display name": "text-<engine>-001",
    },
    "gpt3": {
        "ada": {
            "parameters": 350 * 10**6,
            "size_str": "350m",
            "d_model": 1024,
            "num_layers": 24,
            "d_attn": 16,
            "d_ff": 0,
        },
        "babbage": {
            "parameters": 1.3 * 10**9,
            "size_str": "1.3b",
            "d_model": 2048,
            "num_layers": 24,
            "d_attn": 24,
            "d_ff": 0,
        },
        "curie": {
            "parameters": 6.7 * 10**9,
            "size_str": "6.7b",
            "d_model": 4096,
            "num_layers": 32,
            "d_attn": 32,
            "d_ff": 0,
        },
        "davinci": {
            "parameters": 175 * 10**9,
            "size_str": "175b",
            "d_model": 12288,
            "num_layers": 96,
            "d_attn": 96,
            "d_ff": 0,
        },
        "objective": "LM",
        "training data size": 300 * 10**9,
        "compute": 0,
        "training data": "reddit outbound links with 3+ karma",
        "model display name": "GPT-3",
    },
    "text-davinci-002": {
        "InstructGPT": {
            "parameters": 175 * 10**9,
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
        },
        "objective": "LM",
        "training data size": 300 * 10**9,
        "compute": 0,
        "model display name": "text-davinci-002",
    },
    "text-davinci-003": {
        "InstructGPT": {
            "parameters": 175 * 10**9,
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
        },
        "objective": "LM",
        "training data size": 300 * 10**9,
        "compute": 0,
        "model display name": "text-davinci-003",
    },
    "ChatGPT": {
        "ChatGPT": {
            "parameters": 175 * 10**9,
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
        },
        "objective": "LM",
        "training data size": 300 * 10**9,
        "compute": 0,
        "model display name": "ChatGPT",
    },
    "GPT-4": {
        "GPT-4": {
            "parameters": 175 * 10**9,
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
        },
        "objective": "LM",
        "training data size": 300 * 10**9,
        "compute": 0,
        "model display name": "GPT-4",
    },
    "cohere": {
        "small": {
            "parameters": 409.3 * 10**6,
            "nonembedding-parameters": 409.3 * 10**6 - 51463168,
            "size_str": "409m",
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
        },
        "medium": {
            "parameters": 6.067 * 10**9,
            "nonembedding-parameters": 6.067 * 10**9 - 205852672,
            "size_str": "6b",
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
        },
        "commandmedium": {
            "parameters": 6.067 * 10**9,
            "nonembedding-parameters": 6.067 * 10**9 - 205852672,
            "size_str": "6b",
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
        },
        "large": {
            "parameters": 13.12 * 10**9,
            "nonembedding-parameters": 13.12 * 10**9 - 257315840,
            "size_str": "13b",
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
        },
        "xl": {
            "parameters": 52.4 * 10**9,
            "nonembedding-parameters": 52.4 * 10**9 - 411705344,
            "size_str": "52b",
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
        },
        "commandxl": {
            "parameters": 52.4 * 10**9,
            "nonembedding-parameters": 52.4 * 10**9 - 411705344,
            "size_str": "52b",
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
        },
        "model display name": "Cohere",
    },
    "cohere-command": {
        "commandmedium": {
            "parameters": 6.067 * 10**9,
            "nonembedding-parameters": 6.067 * 10**9 - 205852672,
            "size_str": "6b",
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
        },
        "commandxl": {
            "parameters": 52.4 * 10**9,
            "nonembedding-parameters": 52.4 * 10**9 - 411705344,
            "size_str": "52b",
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
        },
        "command-medium": {
            "parameters": 6.067 * 10**9,
            "nonembedding-parameters": 6.067 * 10**9 - 205852672,
            "size_str": "6b",
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
        },
        "command-xl": {
            "parameters": 52.4 * 10**9,
            "nonembedding-parameters": 52.4 * 10**9 - 411705344,
            "size_str": "52b",
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
        },
        "commandmediumnightly": {
            "parameters": 6.067 * 10**9,
            "nonembedding-parameters": 6.067 * 10**9 - 205852672,
            "size_str": "6b",
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
        },
        "commandxlnightly": {
            "parameters": 52.4 * 10**9,
            "nonembedding-parameters": 52.4 * 10**9 - 411705344,
            "size_str": "52b",
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
        },
        "model display name": "Cohere-command",
    },
    "huggingface": {"bigscience-bloom": {
        "176b": {
            "parameters": 176 * 10**9,
            "nonembedding-parameters": 172.6 * 10**9,
            "size_str": "176b",
            "d_model": 14336,
            "num_layers": 70,
            "d_attn": 112,
            "d_ff": 0,
            "context_window": 2048,
        },
        "560m": {
            "parameters": 560 * 10**6,
            "nonembedding-parameters": 302 * 10**6,
            "size_str": "560m",
            "d_model": 1024,
            "num_layers": 24,
            "d_attn": 16,
            "d_ff": 0,
            "context_window": 0,
        },
        "1b1": {
            "parameters": 1.1 * 10**9,
            "nonembedding-parameters": 680 * 10**6,
            "size_str": "1.1b",
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
            "context_window": 0,
        },
        "1b7": {
            "parameters": 1.7 * 10**9,
            "nonembedding-parameters": 1.2 * 10**9,
            "size_str": "1.7b",
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
            "context_window": 0,
        },
        "3b": {
            "parameters": 3 * 10**9,
            "nonembedding-parameters": 2.4 * 10**9,
            "size_str": "3b",
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
            "context_window": 0,
        },
        "7b1": {
            "parameters": 7.1 * 10**9,
            "nonembedding-parameters": 6 * 10**9,
            "size_str": "7.1b",
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
            "context_window": 0,
        },
        "model display name": "BLOOM",
    },
    "facebook-opt": {
        "6.7b": {
            "parameters": 6.7 * 10**9,
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
            "context_window": 0,
        },
        "13b": {
            "parameters": 13 * 10**9,
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
            "context_window": 0,
        },
        "30b": {
            "parameters": 30 * 10**9,
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
            "context_window": 0,
        },
        "66b": {
            "parameters": 66 * 10**9,
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
            "context_window": 0,
        },
        "175b": {
            "parameters": 175 * 10**9,
            "d_model": 12288,
            "num_layers": 96,
            "d_attn": 96,
            "d_ff": 4 * 12288,
            "context_window": 0,
        },
        "2.7b": {
            "parameters": 2.7 * 10**9,
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
            "context_window": 0,
        },
        "1.3b": {
            "parameters": 1.3 * 10**9,
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
            "context_window": 0,
        },
        "350m": {
            "parameters": 350 * 10**6,
            "d_model": 768,
            "num_layers": 24,
            "d_attn": 12,
            "d_ff": 3072,
            "context_window": 2048,
        },
        "125m": {
            "parameters": 125 * 10**6,
            "d_model": 768,
            "num_layers": 12,
            "d_attn": 12,
            "d_ff": 3072,
            "context_window": 0,
        },
        "model display name": "OPT",
    },
    "google-flan-t5": {
        "xxl": {
            "parameters": 11 * 10**9,
            "d_model": 0,
            "size_str": "11b",
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
            "context_window": 0,
        },
        "large": {"parameters": 780 * 10**6,
              "d_model": 0,
              "num_layers": 0,
              "size_str": "780m",
              "d_attn": 0,
              "d_ff": 0,
              "context_window": 0,},
        "xl": {"parameters": 3 * 10**9,
               "d_model": 0,
               "size_str": "3b",
               "num_layers": 0,
               "d_attn": 0,
               "d_ff": 0,
               "context_window": 0},
        "model display name": "Flan-T5"
     },
    "facebook-blenderbot": {
        "90m": {
            "parameters": 90 * 10**6,
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
            "context_window": 0,
        },
        "3b": {
            "parameters": 3 * 10**9,
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
            "context_window": 0,
        },
        "9b": {
            "parameters": 9 * 10**9,
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
            "context_window": 0,
        },
        "model display name": "BlenderBot",
    },
    "EleutherAI-gpt-j": {
        "6b": {
            "parameters": 6 * 10**9,
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
            "context_window": 0,
        },
        "model display name": "EleutherAI",
    },
    "EleutherAI-gpt-neo": {
        "20b": {
            "parameters": 20 * 10**9,
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
            "context_window": 0,
        },
        "125m": {
            "parameters": 125 * 10**6,
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
            "context_window": 0,
        },
        "1.3b": {
            "parameters": 1.3 * 10**9,
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
            "context_window": 0,
        },
        "2.7b": {
            "parameters": 2.7 * 10**9,
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
            "context_window": 0,
        },
        "model display name": "EleutherAI",
    },
    "EleutherAI-gpt-neox": {
        "20b": {
            "parameters": 20 * 10**9,
            "d_model": 0,
            "num_layers": 0,
            "d_attn": 0,
            "d_ff": 0,
            "context_window": 0,
        },
        "model display name": "EleutherAI",
    },
    "bigscience-T0pp": {
        "11b": {
            "parameters": 11 * 10**9
        },
        "model display name": "T0pp"
    },
    "bigscience-T0": {
        "11b": {
            "parameters": 11 * 10**9
        },
        "3b": {
            "parameters": 3 * 10**9
        },
        "model display name": "T0"
    },
    "roberta": {
        "base": {"parameters": 125 * 10**6,
                 "size_str": "125m"
                 },
        "large": {"parameters": 355 * 10**6,
                  "size_str": "355m"
                  },
        "model display name": "RoBERTa",
    },
    "gpt2": {
        "medium": {"parameters": 354 * 10**6,
                   "size_str": "364m"
                   },
        "large": {"parameters": 774 * 10**6,
                  "size_str": "774m"
                  },
        "xl": {"parameters": 1.6 * 10**9,
               "size_str": "1.6b"
               },
        "model display name": "GPT-2",
    },
    "bert-base": {
        "cased": {"parameters": 110 * 10**6,
                  "size_str": "110m-cased"},
        "uncased": {"parameters": 110 * 10**6,
                    "size_str": "110m"
                    },
        "model display name": "BERT",
    }},
}


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def parse_aggregated_results(file_path):
    with open(file_path, "r") as infile:
        data = json.load(infile)
    arguments = data["config"]
    results = data["results"]
    aggregated_results = {}
    mean_accuracies = []
    templates = []
    templates_d = {}
    for result_d in results:
        if "acc" in result_d:
            mean_accuracies.append(result_d["acc"] * 100.)
            templates.append(result_d["prompt_name"])
            templates_d[f'prompt_{result_d["prompt_name"]}'] = result_d["acc"] * 100.
    mean_accuracy = np.mean(mean_accuracies)
    std = np.std(mean_accuracies)
    aggregated_results["mean_accuracy"] = mean_accuracy
    aggregated_results["std"] = std
    aggregated_results["template_results"] = templates_d
    return aggregated_results, arguments


def parse_examples_file(file_path):
    with open(file_path, "r") as infile:
        data = json.load(infile)
    print()


def parse_slim_file(file_path):
    with open(file_path, "r") as infile:
        data = json.load(infile)
    template_results = {}
    for result in data["results"]:
        if "acc" in result:
            template_results[f"prompt_{result['prompt_name']}"] = result["acc"] * 100.
            k = result["dataset_name"].split("-")[0]
    return template_results, k


def parse_huggingface_file(file):
    with open(file, 'r') as json_file:
        json_list = list(json_file)

    arguments = {
        "type_implicature": "particularised",
        "test_input_data_path": "data/test_conversational_implicatures.csv",
        "dev_input_data_path": "data/dev_conversational_implicatures.csv",
        "seed": 0
    }
    # Take out all the test data that we need in the test set.
    dataset = load_datasets(dotdict(arguments))
    test_data = dataset._data["test_data"]

    parsed_result = []
    parsed_doc_idx = []
    parsed_template_nums = []
    all_results = []
    example_results = defaultdict(lambda: defaultdict(dict))
    iterator = dataset.get_implicature_iterator(k_shot=int(json.loads(json_list[0])["dataset_name"].split("-")[0]))
    prompt_examples_per_idx = [example_with_prompts for example_with_prompts in iterator]
    for json_str in json_list:
        result = json.loads(json_str)
        parsed_result.append(result["ctx"])
        parsed_template_nums.append(result["prompt_name"])
        parsed_doc_idx.append(result["doc_id"])
        original_example = test_data[result["doc_id"]]
        example_with_prompts = prompt_examples_per_idx[result["doc_id"]]
        assert original_example == example_with_prompts["example"]
        prompt_examples = example_with_prompts["prompts"]
        for prompt_example in prompt_examples:
            for key in ["utterance", "response"]:
                assert prompt_example[key].replace("\r", "") in result[
                    "ctx"].replace("\r", ""), "Mismatch between lm-eval doc_idx and original example test set."

        for key in ["utterance", "response"]:
            assert original_example[key].replace("\r", "") in result["ctx"].replace("\r", ""), "Mismatch between lm-eval doc_idx and original example test set."
        all_results.append(result)
        example_results[f"prompt_{result['prompt_name']}"][result["doc_id"]] = {"id": result["doc_id"],
                                                                                "original_example": original_example,
                                                                                "true": result["target"],
                                                                                "pred": result["pred"],
                                                                                "correct": int(result["target"] == result["pred"]),
                                                                                "prompt_examples": prompt_examples}
    sorted_results = [(y, x) for _, x, y in sorted(zip(parsed_doc_idx, parsed_result, parsed_template_nums))]
    sorted_template_nums = [x for _, x in sorted(zip(parsed_doc_idx, parsed_template_nums))]
    return sorted_results, sorted_template_nums, all_results, example_results


def convert_lm_eval_results(results_folder: str, output_folder):
    aggregated_results_prefix = "agg"
    predictions_prefix = "examples"
    acc_per_template_prefix = "slim"

    result_analysis = defaultdict(lambda: defaultdict(dict))
    other_file_results = defaultdict(lambda: defaultdict(dict))
    for filename in os.listdir(results_folder):
        print(filename)
        file_details = filename.split("_")[0].split("-")
        file_type = file_details[0]
        model_id = "-".join(file_details[1:-1])
        pars = file_details[-1]
        k = filename.split("-shot")[0].split("-")[-1]
        file_path = os.path.join(results_folder, filename)
        if file_type == aggregated_results_prefix:
            aggregated_results, arguments = parse_aggregated_results(file_path)
            result_analysis[k][f"{model_id}-{pars}"].update(aggregated_results)
            if k in other_file_results:
                if f"{model_id}-{pars}" in other_file_results[k]:
                    assert other_file_results[k][f"{model_id}-{pars}"]["template_results"] == aggregated_results["template_results"]
            result_analysis[k][f"{model_id}-{pars}"]["arguments"] = arguments
        elif file_type == predictions_prefix:
            _, _, predictions, example_results = parse_huggingface_file(file_path)
            result_analysis[k][f"{model_id}-{pars}"]["example_results"] = example_results
        elif file_type == acc_per_template_prefix:
            template_results, k = parse_slim_file(file_path)
            if "template_results" in result_analysis[k][f"{model_id}-{pars}"]:
                assert result_analysis[k][f"{model_id}-{pars}"]["template_results"] == template_results
            else:
                other_file_results[k][f"{model_id}-{pars}"]["template_results"] = template_results
        elif file_type == "emissions":
            continue
        else:
            continue

    # Write results for all models to a file
    save_results(output_folder, result_analysis)


def read_results_file_from_folder(results_folder):
    """Assumes file that starts with 'results' is the one to read. Only works if the folder has one file like this."""
    files_in_dir = [
        file for file in os.listdir(results_folder) if file.startswith("results")
    ]
    assert (
        len(files_in_dir) == 1
    ), f"Found {len(files_in_dir)} result files in {results_folder} instead of 1."
    results_file = files_in_dir.pop()
    results_file = os.path.join(results_folder, results_file)
    with open(results_file, "r") as infile:
        results = json.load(infile)
    return results


def error_analysis_per_k(results_folder, output_folder):

    results = read_results_file_from_folder(results_folder)

    # Extract the models that are part of this results file and the k-shot argument.
    model_ids = list(results.keys())
    model_ids.remove("predictions")
    model_ids.remove("arguments")
    k_shot = str(results["arguments"]["k_shot"])

    write_model_ids_table = {
        "cohere-commandmedium": "cohere-command-medium",
        "cohere-commandxl": "cohere-command-xl",
        "openai-ada": "openai-ada",
        "openai-babbage": "openai-babbage",
        "openai-chatgpt": "openai-chatgpt",
        "openai-curie": "openai-curie",
        "openai-davinci": "openai-davinci",
        "openai-gpt4": "openai-gpt4",
        "openai-textada001": "openai-text-ada-001",
        "openai-textbabbage001": "openai-text-babbage-001",
        "openai-textcurie001": "openai-text-curie-001",
        "openai-textdavinci001": "openai-text-davinci-001",
        "openai-textdavinci002": "openai-text-davinci-002",
        "openai-textdavinci003": "openai-text-davinci-003",
    }
    write_model_ids = [write_model_ids_table[model_id] for model_id in model_ids]

    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    # Hardcode stuff
    translate_yes_to_no = {"yes": "no", "no": "yes"}
    template_keys_translation = {
        "prompt_template_0": "prompt_template_1",
        "prompt_template_1": "prompt_template_2",
        "prompt_template_2": "prompt_template_3",
        "prompt_template_3": "prompt_template_4",
        "prompt_template_4": "prompt_template_5",
        "prompt_template_5": "prompt_template_6",
    }

    # Loop over all the models and save results per k.
    result_analysis = {k_shot: {}}
    for i, model_id in enumerate(model_ids):
        model_results_folder = os.path.join(output_folder, model_id)
        if not os.path.exists(model_results_folder):
            os.mkdir(model_results_folder)
        model_results_folder = os.path.join(output_folder, model_id, str(k_shot))
        if not os.path.exists(model_results_folder):
            os.mkdir(model_results_folder)

        # Extract the accuracy per prompt template
        template_results = {}
        for key, values in results[model_id].items():
            if "prompt_template_" in key:
                template_results[template_keys_translation[key]] = values[
                    "implicature_metrics"
                ]

        # Save all the results
        model_results = {
            "mean_accuracy": results[model_id]["mean_accuracy"],
            "std": results[model_id]["std"],
            "template_results": template_results,
        }
        result_analysis[k_shot][write_model_ids[i]] = model_results
    example_results_per_model = {
        model_id: defaultdict(lambda: defaultdict(dict)) for model_id in model_ids
    }
    for i, predictions_per_model in enumerate(results["predictions"]):
        model_id = list(predictions_per_model.keys())[0]
        assert model_id in model_ids, f"Unknown model id {model_id}"
        model_results = predictions_per_model[model_id]
        original_example_id = i
        original_example = predictions_per_model["original_example"]
        prompt_examples = predictions_per_model["prompt_examples"]
        true = predictions_per_model["original_example"]["implicature"]
        for prompt_template in template_keys_translation.keys():
            if prompt_template not in model_results:
                continue
            example_correct = model_results[prompt_template]["implicature_result"][
                "example_correct"
            ]
            pred = true if example_correct else translate_yes_to_no[true]
            example_results_per_model[model_id][
                template_keys_translation[prompt_template]
            ][i] = {
                "id": original_example_id,
                "original_example": original_example,
                "true": true,
                "pred": pred,
                "correct": int(example_correct),
                "prompt_examples": prompt_examples,
            }
    for i, model_id in enumerate(example_results_per_model.keys()):
        result_analysis[k_shot][write_model_ids[i]][
            "example_results"
        ] = example_results_per_model[model_id]
    # Write results for all models to a file
    save_results(output_folder, result_analysis)


def save_results(save_folder, results):
    results_path = os.path.join(save_folder, "all_results.json")
    if os.path.exists(results_path):
        with open(results_path, "r") as infile:
            existing_results = json.load(infile)
    else:
        existing_results = {}

    for k, results_per_k in results.items():
        if k not in existing_results:
            existing_results[k] = {}
        for model, model_results in results_per_k.items():
            if model in existing_results[k]:
                print(f"WARNING: overwriting results for {model} {k}-shot")
            existing_results[k][model] = model_results
    with open(results_path, "w") as outfile:
        json.dump(existing_results, outfile, indent=4)
    return results_path


def par_calc(d_model, num_layers, d_attn, d_ffn):
    return 2 * d_model * num_layers * (2 * d_attn + d_ffn)


def get_nonembedding_pars_s2s(hf_name):
    config = AutoConfig.from_pretrained(hf_name)
    d_model = config.hidden_size
    encoder_pars = par_calc(
        d_model,
        config.encoder_layers,
        config.encoder_attention_heads,
        config.encoder_ffn_dim,
    )
    decoder_pars = par_calc(
        d_model,
        config.decoder_layers,
        config.decoder_attention_heads,
        config.decoder_ffn_dim,
    )
    return encoder_pars + decoder_pars


def get_nonembedding_pars_openai(engine):
    d_model = NUM_PARAMETERS["gpt3"][engine]["d_model"]
    num_layers = NUM_PARAMETERS["gpt3"][engine]["num_layers"]
    d_attn = NUM_PARAMETERS["gpt3"][engine]["d_attn"]
    d_ffn = 4 * d_model
    return 2 * d_model * num_layers * (2 * d_attn + d_ffn)


def get_nonembedding_pars(hf_name):
    hf_name = hf_name.split("-")
    hf_name = f"{hf_name[0]}/{'-'.join(hf_name[1:])}"
    config = AutoConfig.from_pretrained(hf_name)
    if "blenderbot" in hf_name.lower():
        return get_nonembedding_pars_s2s(hf_name)
    if "opt-175b" in hf_name.lower():
        return get_nonembedding_pars_openai("text-davinci-001")
    d_model = config.hidden_size
    num_layers = config.num_hidden_layers
    d_attn = config.num_attention_heads
    if "gpt-j-6B" in hf_name:
        d_ffn = 16384
    elif "gpt-neo" in hf_name:
        d_ffn = 4 * d_model
    else:
        d_ffn = config.ffn_dim
    return 2 * d_model * num_layers * (2 * d_attn + d_ffn)


def get_nonembedding_pars_T0(hf_name):
    config = AutoConfig.from_pretrained(hf_name)
    d_model = config.hidden_size
    encoder_pars = par_calc(d_model, config.num_layers, config.num_attention_heads, 4 * d_model)
    decoder_pars = par_calc(d_model, config.num_layers, config.num_attention_heads, 4 * d_model)
    return encoder_pars + decoder_pars


def parse_model_id(model_id: str):
    """
    Deal with the unsystematic way of identifying models by HuggingFace and methods to calculate the number
    of non-embedding parameters.
    :param model_id:
    :return:
    """
    model = "-".join(model_id.split("-")[1:])
    if model == "text-davinci-002" or model == "textdavinci002":
        model_class = "text-davinci-002"
        model = "davinci002"
    elif model == "textdavinci003" or model == "text-davinci-003":
        model_class = "text-davinci-003"
        model = "davinci003"
    elif model == "chatgpt":
        model_class = "ChatGPT"
        model = "chatgpt"
    elif model == "gpt4":
        model_class = "GPT-4"
        model = "GPT-4"
    elif "text" in model:
        model_class = "text-<engine>-001"
    elif "openai" in model_id:
        model_class = "gpt3"
    elif "cohere" in model_id.lower() and "command" not in model_id:
        model_class = "cohere"
    elif "cohere" in model_id.lower() and "command" in model_id:
        model_class = "cohere-command"
    else:
        model_class = "huggingface"
        model = model_id

    if model == "davinci001" or model == "davinci002" or model == "davinci003" or model == "chatgpt" or model == "GPT-4":
        # Don't know the size.
        return 0, "unknown", model_class

    if "text" in model:
        engine = model.split("-")[-2]
        return 0, engine, model_class

    if model_class == "huggingface":
        model_identifier = "-".join(model.split("-")[:-1])
        model_size_id = model.split("-")[-1].lower()
        if model_identifier == "facebook" and model_size_id == "blenderbot":
            model_identifier = "facebook-blenderbot"
            model_size_id = "90m"
            model = "facebook-blenderbot-90M"
        if "blenderbot-9b" in model.lower():
            model_identifier = "facebook-blenderbot"
            model_size_id = "9b"
            model = "hyunwoongko-blenderbot-9B"
        if model_identifier == "bigscience":
            model_identifier = "bigscience-bloom"
            model_size_id = "176b"
            model = "bigscience-bloom"
        if "T0pp" in model_id:
            model_identifier = "bigscience-T0pp"
            model_size_id = "11b"
            model = "bigscience/T0pp"
            return get_nonembedding_pars_T0(model), model_size_id, NUM_PARAMETERS["huggingface"][model_identifier]["model display name"]
        if "flan" in model_id.lower():
            model_identifier = "google-flan-t5"
            model_size_id = model_id.split("-")[-1]
            model = f"google/flan-t5-{model_size_id}"
            model_size_id = NUM_PARAMETERS["huggingface"][model_identifier][model_size_id]["size_str"]
            return get_nonembedding_pars_T0(model), model_size_id, NUM_PARAMETERS["huggingface"][model_identifier][
                "model display name"]
        if "T0" in model_id and "pp" not in model_id:
            model_identifier = "bigscience-T0"
            if "3b" in model_id.lower():
                model_size_id = "3b"
                model = "bigscience/T0_3B"
            else:
                model_size_id = "11b"
                model = "bigscience/T0"
            return get_nonembedding_pars_T0(model), model_size_id, NUM_PARAMETERS["huggingface"][model_identifier]["model display name"]
        if "opt-175b" in model.lower():
            return get_nonembedding_pars_openai("davinci"), "175b", NUM_PARAMETERS["huggingface"][model_identifier]["model display name"]
        if "nonembedding-parameters" in NUM_PARAMETERS["huggingface"][model_identifier][model_size_id].keys():
            nonembedding_parameters = NUM_PARAMETERS["huggingface"][model_identifier][model_size_id][
                "nonembedding-parameters"
            ]
        else:
            if "bert" in model_id or "gpt2" in model_id:
                # Don't know the nonembedding size.
                return 0, NUM_PARAMETERS["huggingface"][model_identifier][model_size_id]["size_str"], NUM_PARAMETERS["huggingface"][model_identifier]["model display name"]
            nonembedding_parameters = get_nonembedding_pars(model)
        name = NUM_PARAMETERS["huggingface"][model_identifier]["model display name"]
    elif "nonembedding-parameters" in NUM_PARAMETERS[model_class][model].keys():
        nonembedding_parameters = NUM_PARAMETERS[model_class][model][
            "nonembedding-parameters"
        ]
        model_size_id = NUM_PARAMETERS[model_class][model][
            "size_str"
        ]
        name = NUM_PARAMETERS[model_class]["model display name"]
    else:
        nonembedding_parameters = get_nonembedding_pars_openai(model.replace("text-", "").replace("-001", ""))
        model_size_id = NUM_PARAMETERS["gpt3"][model.replace("text-", "").replace("-001", "")]["size_str"]
        name = NUM_PARAMETERS[model_class]["model display name"]
    return nonembedding_parameters, model_size_id, name


def plot_scale_graph(results_path, models_to_show, label_order, normalize_metric=False):
    with open(results_path, "r") as infile:
        results = json.load(infile)

    if normalize_metric:
        human_avg = renormalize(HUMAN_AVG_PERFORMANCE)
        human_best = renormalize(HUMAN_BEST_PERFORMANCE)
    else:
        human_avg, human_best = HUMAN_AVG_PERFORMANCE, HUMAN_BEST_PERFORMANCE
    results_folder = results_path.split("/")[0]
    k_shot = [int(k) for k in list(results.keys())]
    lines = {}
    model_classes = []
    colors_counter = -1
    model_colors = plt.cm.Dark2(np.linspace(0, 1, 10))
    for k in k_shot:
        data = results[str(k)]
        if k not in lines:
            lines[k] = {}
        for model_id in data.keys():
            if "cohere-command" in model_id and "nightly" in model_id:
                continue
            model_size, _, name = parse_model_id(model_id)
            if name not in models_to_show:
                continue
            if name not in lines[k]:
                lines[k][name] = {}
                lines[k][name]["mean_line"] = {"x": [], "y": [], "std": []}
            model_classes.append(name)
            mean_accuracy = data[model_id]["mean_accuracy"]
            if normalize_metric:
                mean_accuracy = renormalize(mean_accuracy)
            std = data[model_id]["std"]
            lines[k][name]["mean_line"]["x"].append(model_size)
            lines[k][name]["mean_line"]["y"].append(mean_accuracy)
            lines[k][name]["mean_line"]["std"].append(std)
            lines[k][name]["name"] = name

    linewidth = 3

    markersize = 10
    # Mean accuracy plot per model of accuracy v. k
    model_classes = list(set(model_classes))
    plt.figure(figsize=(16, 14), dpi=200)
    x_min = float("inf")
    x_max = 0
    for i, model_class in enumerate(model_classes):
        linestyle = "dashed"
        for j, k in enumerate(k_shot):
            # color = k_colors[j]
            line_color = MODEL_COLORS[model_class]
            x = lines[k][model_class]["mean_line"]["x"]
            name = lines[k][model_class]["name"]
            y = [y for _, y in sorted(zip(x, lines[k][model_class]["mean_line"]["y"]))]
            std = [x for _, x in sorted(zip(x, lines[k][model_class]["mean_line"]["std"]))]
            x = sorted(x)
            if x[0] < x_min:
                x_min = x[0]
            if x[-1] > x_max:
                x_max = x[-1]
            plt.errorbar(x, y,
                         yerr=std,
                         label=f"{name}-{k}-shot", marker="o", linestyle=linestyle, color=line_color,
                         markersize=markersize, linewidth=linewidth)
    plt.hlines(y=human_avg, xmin=x_min, xmax=x_max, label="Avg. human performance", linestyles="solid",
               color="grey", linewidth=linewidth)
    plt.hlines(y=human_best, xmin=x_min, xmax=x_max, label="Best human", linestyles="solid",
               color="black", linewidth=linewidth)
    if not normalize_metric:
        plt.hlines(y=50.0, xmin=x_min, xmax=x_max, label="Random chance", linestyles="dotted", color="red")
    plt.legend(loc="upper left", fontsize=14)
    # plt.xticks(sorted_x, sorted_x)  # Set text labels.
    plt.ylim(bottom=0., top=100.0)
    plt.xscale("log")
    plt.xlabel("Model Parameters (Non-Embedding)", fontsize=14)
    ylabel = "Accuracy (%)"
    title = "Accuracy (%) vs. model size"
    if normalize_metric:
        ylabel = "Normalized " + ylabel
        title = "Normalized " + title
    plt.ylabel(ylabel, fontsize=14)
    plt.title(title, fontsize=18)
    plt.savefig(os.path.join(results_folder, f"accuracy_v_size.png"))
    # different plot per k-shot in subplots graph
    # fig, axs = plt.subplots(3, 2, sharey=True, sharex=True, figsize=(16, 18), dpi=600)
    fig, axs = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(28, 14), dpi=600)
    # model_colors = plt.cm.Dark2(np.linspace(0, 1, len(model_classes)))
    legend_lines, legend_labels = [], []
    x_min = float("inf")
    x_max = 0
    adjusted_k_shot = [0, 5]
    k_to_str = {0: "Zero",
                1: "One",
                5: "Five",
                10: "Ten",
                15: "Fifteen",
                30: "Thirty"}
    legend_d = {}
    legend_lines, legend_labels = [], []
    for i, k in enumerate(adjusted_k_shot):
        # ax = axs[int(i // 2), i % 2]
        ax = axs[i % 2]
        for j, model_class in enumerate(list(model_classes)):
            color = MODEL_COLORS[model_class]
            x = lines[k][model_class]["mean_line"]["x"]
            name = lines[k][model_class]["name"]
            y = [y for _, y in sorted(zip(x, lines[k][model_class]["mean_line"]["y"]))]
            std = [x for _, x in sorted(zip(x, lines[k][model_class]["mean_line"]["std"]))]
            x = sorted(x)
            if x[0] < x_min:
                x_min = x[0]
            if x[-1] > x_max:
                x_max = x[-1]
            line = ax.errorbar(x, y,
                               yerr=std,
                               label=f"{name}-{k}-shot", marker="o", linestyle=LINESTYLE_GROUPING[name], color=color,
                               markersize=markersize, linewidth=linewidth)
            if k == 0:
                humanline = ax.hlines(y=human_avg, xmin=x_min, xmax=x_max, label="Avg. human", linestyles="solid",
                           color="grey", linewidth=linewidth)
                humanlinebest = ax.hlines(y=human_best, xmin=x_min, xmax=x_max, label="Best human",
                                      linestyles="solid",
                                      color="black", linewidth=linewidth)
            # if not normalize_metric and k == 0:
            #     randomline = ax.hlines(y=50.0, xmin=x_min, xmax=x_max, label="Random chance", linestyles="dotted", color="red", linewidth=linewidth)
            if not normalize_metric:
                randomline = ax.hlines(y=50.0, xmin=x_min, xmax=x_max, label="Random chance", linestyles="solid",
                                       color="red", linewidth=linewidth)
            if i == 0:
                legend_lines.append(line)
                legend_labels.append(f"{name}")
                legend_d[f"{name}"] = line
            # ax.set_xticks(sorted_x, sorted_x)  # Set text labels.
            plt.ylim(bottom=0., top=100.0)
            ax.set_xscale("log")
            ax.title.set_text(f"{k_to_str[k]}-shot.")
            ax.title.set_size(50)
            if i % 2 == 0:
                ylabel = "Accuracy (%)"
                if normalize_metric:
                    ylabel = "Normalized " + ylabel
                ax.set_ylabel(ylabel, fontsize=50)
            ax.xaxis.set_tick_params(labelsize=30)
            ax.yaxis.set_tick_params(labelsize=30)
            # ax.set_xticks(fontsize=18)
            if int(i // 2) == 0:
                ax.set_xlabel("Model Parameters (Non-Embedding)", fontsize=50)
    plt.ylim(bottom=0., top=100.0)
    legend_lines.append(humanline)
    legend_labels.append(f"Avg. human")
    legend_d["Avg. human"] = humanline
    legend_lines.append(humanlinebest)
    legend_labels.append(f"Best human")
    legend_d["Best human"] = humanlinebest
    if not normalize_metric:
        legend_lines.append(randomline)
        legend_labels.append(f"Random chance")
        legend_d["Random chance"] = randomline
    if not normalize_metric:
        loc = "lower center"
    else:
        loc = "upper right"
    # fig.legend(legend_lines, legend_labels, fontsize=18)
    ordered_legend_lines = [legend_d[line_n] for line_n in label_order]
    fig.legend(ordered_legend_lines, label_order, fontsize=43, bbox_to_anchor=[0.5, 0.1], loc=loc, ncol=3)
    plt.suptitle(f"Performance on the implicature task \nfor k-shot evaluations", fontsize=50)
    # plt.tight_layout()
    # plt.subplots_adjust(bottom=0.1, top=0.9, right=0.9, left=0.1)
    plt.subplots_adjust(top=0.75)
    plt.tight_layout()
    plt.savefig(os.path.join(results_folder, f"accuracy_v_size_subplots.png"))
    plt.clf()

    x_min = float("inf")
    x_max = 0
    k_to_str = {0: "Zero",
                1: "One",
                5: "Five",
                10: "Ten",
                15: "Fifteen",
                30: "Thirty"}
    for i, k in enumerate(k_shot):
        legend_lines, legend_labels = [], []
        legend_d = {}
        plt.figure(figsize=(16, 14), dpi=200)
        for j, model_class in enumerate(list(model_classes)):
            color = MODEL_COLORS[model_class]
            x = lines[k][model_class]["mean_line"]["x"]
            name = lines[k][model_class]["name"]
            y = [y for _, y in sorted(zip(x, lines[k][model_class]["mean_line"]["y"]))]
            std = [x for _, x in sorted(zip(x, lines[k][model_class]["mean_line"]["std"]))]
            x = sorted(x)
            if x[0] < x_min:
                x_min = x[0]
            if x[-1] > x_max:
                x_max = x[-1]
            line = plt.errorbar(x, y,
                                yerr=std,
                                label=f"{model_class}-{k}-shot", marker="o", color=color,
                                linestyle=LINESTYLE_GROUPING[name],
                                markersize=markersize, linewidth=linewidth)
            if k == 0:
                humanline = plt.hlines(y=human_avg, xmin=x_min, xmax=x_max, label="Avg. human",
                                      linestyles="solid",
                                      color="grey", linewidth=linewidth)
                humanlinebest = plt.hlines(y=human_best, xmin=x_min, xmax=x_max,
                                          label="Best human",
                                          linestyles="solid",
                                          color="black", linewidth=linewidth)
            legend_lines.append(line)
            legend_labels.append(f"{name}")
            legend_d[f"{name}"] = line
        if not normalize_metric:
            randomline = plt.hlines(y=50.0, xmin=x_min, xmax=x_max, label="Random chance", linestyles="solid",
                                    color="red", linewidth=linewidth)
        # ax.set_xticks(sorted_x, sorted_x)  # Set text labels.
        plt.ylim(bottom=0., top=100.0)
        plt.xscale("log")
        plt.xticks(fontsize=28)
        plt.yticks(fontsize=28)
        # plt.title(f"{k}-shot", fontsize=24)
        plt.xlabel("Model Parameters (Non-Embedding)", fontsize=32)
        ylabel = "Accuracy (%)"
        if normalize_metric:
            ylabel = "Normalized " + ylabel
        plt.ylabel(ylabel, fontsize=32)
        legend_lines.append(humanline)
        legend_labels.append(f"Avg. human")
        legend_d["Avg. human"] = humanline
        legend_lines.append(humanlinebest)
        legend_labels.append(f"Best human")
        legend_d["Best human"] = humanlinebest
        if not normalize_metric:
            legend_lines.append(randomline)
            legend_labels.append(f"Random chance")
            legend_d["Random chance"] = randomline
        if not normalize_metric:
            loc = "lower center"
        else:
            loc = "upper right"
        ordered_legend_lines = [legend_d[line_n] for line_n in label_order]
        plt.legend(ordered_legend_lines, label_order, fontsize=27, loc=loc, ncol=3)
        plt.title(f"{k_to_str[k]}-shot accuracy on the implicature task.", fontsize=32)
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.1, top=0.9, right=0.9, left=0.1)
        plt.savefig(os.path.join(results_folder, f"accuracy_v_size_k={k}.png"))
        plt.clf()


def renormalize(number, random=50.):
    return ((number - random) / random) * 100.


def plot_all_lines(results_path, renormalize_metric=False, use_differences=True):
    linewidth = 3
    markersize = 10

    with open(results_path, "r") as infile:
        results = json.load(infile)
    results_folder = results_path.split("/")[0]
    k_shot = [int(k) for k in list(results.keys())]

    if renormalize_metric:
        human_avg = renormalize(HUMAN_AVG_PERFORMANCE)
        human_best = renormalize(HUMAN_BEST_PERFORMANCE)
    else:
        human_avg, human_best = HUMAN_AVG_PERFORMANCE, HUMAN_BEST_PERFORMANCE

    lines = {}
    model_ids = []
    model_ids_to_plot = ["openai-text-davinci-001",
                         "cohere-commandxl", "facebook-opt-175b"
                         ]
    names = ["text-<engine>-001", "Cohere-command", "OPT"]
    classes = ["text-<engine>-001", "Cohere-command", "OPT"]
    sizes = ["Unknown", "52B", "175B"]
    zero_shot_per_model = {}
    zero_shot_per_model_per_template = defaultdict(dict)
    for k in k_shot:
        data = results[str(k)]
        for model_id in data.keys():
            model_size, _, name = parse_model_id(model_id)
            if model_id not in lines:
                lines[model_id] = {}
                lines[model_id]["mean_line"] = {"x": [], "y": [], "std": []}
                lines[model_id]["name"] = name
            model_ids.append(model_id)
            mean_accuracy = data[model_id]["mean_accuracy"]
            if k == 0:
                zero_shot_per_model[model_id] = mean_accuracy
            if renormalize_metric:
                mean_accuracy = renormalize(mean_accuracy)
            if use_differences:
                mean_accuracy = mean_accuracy - zero_shot_per_model[model_id]
            std = data[model_id]["std"]
            lines[model_id]["mean_line"]["x"].append(k)
            lines[model_id]["mean_line"]["y"].append(mean_accuracy)
            lines[model_id]["mean_line"]["std"].append(std)
            results_per_line = data[model_id]["template_results"]
            for i, template in enumerate(results_per_line.keys()):
                if i + 1 not in lines[model_id]:
                    lines[model_id][i + 1] = {"x": [], "y": []}
                lines[model_id][i + 1]["x"].append(k)
                y_result = results_per_line[template]
                if k == 0:
                    zero_shot_per_model_per_template[model_id][template] = y_result
                if not use_differences:
                    lines[model_id][i + 1]["y"].append(results_per_line[template])
                else:
                    lines[model_id][i + 1]["y"].append(
                        results_per_line[template] - zero_shot_per_model_per_template[model_id][template])

    # different plot per model to plot with template breakdown
    fig, axs = plt.subplots(1, 3, sharey=True, sharex=True, figsize=(18, 8), dpi=600)
    legend_lines, legend_labels = [], []
    for i, model_id in enumerate(list(model_ids_to_plot)):
        # ax = axs[int(i // 2), i % 2]
        ax = axs[i]
        for line_key, line_data in lines[model_id].items():
            if line_key != "mean_line" and line_key != "name":
                prompt_group = PROMPT_GROUPING[f"prompt_template_{line_key}"]
                linestyle = GROUP_LINESTYLE[prompt_group]
                line, = ax.plot(line_data["x"], line_data["y"], label=f"Prompt template {line_key}", marker="o",
                                linestyle=linestyle, markersize=markersize, linewidth=linewidth)
                if i == 1:
                    legend_lines.append(line)
                    legend_labels.append(f"Prompt template {line_key}")
        if i == 0:
            ax.legend(handles=legend_lines, labels=legend_labels, fontsize=17, loc="upper center")
        ax.set_xticks(k_shot, k_shot)
        ax.title.set_text(f"{names[i]}-{sizes[i]}")
        ax.title.set_size(16)
        if i == 0:
            ylabel = "Accuracy (%)"
            if renormalize_metric:
                ylabel = "Normalized " + ylabel
            if use_differences:
                ylabel = "Relative Accuracy (%)"
            ax.set_ylabel(ylabel, fontsize=24)
        ax.xaxis.set_tick_params(labelsize=18)
        ax.yaxis.set_tick_params(labelsize=20)
        ax.set_xlabel("k-shot", fontsize=24)
    ymin = 50.
    ymax = 100.
    if use_differences:
        ymin = -10
        ymax = 20
    plt.ylim(bottom=ymin, top=ymax)
    if not use_differences:
        plt.suptitle(f"Relative accuracy (w.r.t 0-shot) on the implicature task for the largest model of each "
                     "model class \nfor all prompt templates.", fontsize=28)
    else:
        plt.suptitle(
            f"Relative accuracy (w.r.t. 0-shot) due to in-context examples for all prompt templates.",
            fontsize=24)
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.savefig(os.path.join(results_folder, f"accuracy_v_k_subplots.png"))
    plt.clf()

    # Different multiline plot per model of accuracy v. k for each prompt template
    model_ids = set(model_ids)
    for model_id in list(model_ids):
        plt.figure(figsize=(16, 14), dpi=200)
        for line_key, line_data in lines[model_id].items():
            if line_key != "mean_line" and line_key != "name":
                plt.plot(line_data["x"], line_data["y"], label=f"Prompt template {line_key}", marker="o",
                         linestyle="dashed", markersize=markersize, linewidth=linewidth)
        plt.hlines(y=human_avg, xmin=k_shot[0], xmax=k_shot[-1], label="Avg. human", linestyles="solid",
                   color="grey", linewidth=linewidth)
        plt.hlines(y=human_best, xmin=k_shot[0], xmax=k_shot[-1], label="Best human", linestyles="solid",
                   color="black", linewidth=linewidth)
        if not renormalize_metric:
            plt.hlines(y=50.0, xmin=k_shot[0], xmax=k_shot[-1], label="Random chance", linestyles="dotted", color="red",
                       linewidth=linewidth)
        plt.legend(loc="lower right", fontsize=14)
        plt.xticks(k_shot, k_shot)  # Set text labels.
        if not renormalize_metric:
            plt.ylim(bottom=0, top=100.0)
        plt.xlabel("k-shot", fontsize=14)
        plt.ylabel("Accuracy (%)", fontsize=14)
        plt.title(f"{model_id}\nAccuracy (%) vs. k-shot for each prompt template.", fontsize=18)
        plt.savefig(os.path.join(results_folder, f"{model_id}_accuracy_v_k_per_template.png"))

    # Mean accuracy plot per model of accuracy v. k
    plt.figure(figsize=(16, 14), dpi=200)
    for i, model_id in enumerate(list(model_ids_to_plot)):
        name = names[i]
        color = MODEL_COLORS[classes[i]]
        if name == "InstructGPT":
            zorder = 10
        else:
            zorder = None
        if name == "Cohere":
            zorder = 15
        plt.errorbar(lines[model_id]["mean_line"]["x"], lines[model_id]["mean_line"]["y"],
                     yerr=lines[model_id]["mean_line"]["std"], color=color, zorder=zorder,
                     label=f"{lines[model_id]['name']}-{sizes[i]}", marker="o", linestyle="dashed",
                     markersize=markersize, linewidth=linewidth)
    if not renormalize_metric and not use_differences:
        plt.hlines(y=50.0, xmin=k_shot[0], xmax=k_shot[-1], label="Random chance", linestyles="dotted", color="red",
                   linewidth=linewidth)
    plt.legend(loc="lower right", fontsize=32)
    plt.xticks(k_shot, k_shot, fontsize=28)
    plt.yticks(fontsize=24)# Set text labels.
    ymin = 50.
    ymax = 100.
    if use_differences:
        ymin = -5
        ymax = 17
    plt.ylim(bottom=ymin, top=ymax)
    plt.xlabel("In-context examples (k)", fontsize=32)
    plt.ylabel("Relative Accuracy (%)", fontsize=32)
    plt.subplots_adjust(bottom=0.1, top=0.9, right=0.9, left=0.1)
    if not use_differences:
        plt.title(f"Relative accuracy (w.r.t 0-shot) due to in-context examples.", fontsize=32)
    else:
        plt.title(f"Relative accuracy (w.r.t 0-shot) due to in-context examples.", fontsize=32)
    plt.savefig(os.path.join(results_folder, f"accuracy_v_k.png"))


def plot_few_shot(results_path, models_to_show, label_order, groups_per_model,
                  normalize_metric=False, use_differences=False):
    with open(results_path, "r") as infile:
        results = json.load(infile)

    results_folder = results_path.split("/")[0]
    k_shot = [int(k) for k in list(results.keys())]
    # k_shot = [5]  # TODO: change once all results in
    unique_groups = set(groups_per_model)
    model_to_group = {model: group for model, group in zip(models_to_show, groups_per_model)}
    lines = {}
    model_classes = []
    ids_per_class = {}
    model_ids = []
    zero_shot_per_model = {}
    zero_shot_per_model_per_template = defaultdict(dict)
    for k in k_shot:
        data = results[str(k)]
        for model_id in data.keys():
            model_size, size_id, name = parse_model_id(model_id)
            if name in models_to_show:
                group = model_to_group[name]
                if group not in lines:
                    lines[group] = {}
                    lines[group]["mean_line"] = {"x": [], "y": [], "std": []}
                    lines[group]["name"] = name
                    lines[group]["size"] = size_id
            if model_id not in lines:
                lines[model_id] = {}
                lines[model_id]["mean_line"] = {"x": [], "y": [], "std": []}
                lines[model_id]["name"] = name
                lines[model_id]["size"] = size_id
            model_ids.append(model_id)
            model_classes.append(name)
            if name not in ids_per_class:
                ids_per_class[name] = {"model_ids": [], "scores": []}
            mean_accuracy = data[model_id]["mean_accuracy"]
            if k == 0:
                zero_shot_per_model[model_id] = mean_accuracy
            if k == 5:
                ids_per_class[name]["model_ids"].append(model_id)
                ids_per_class[name]["scores"].append(mean_accuracy)
            if normalize_metric:
                mean_accuracy = renormalize(mean_accuracy)
            if use_differences:
                mean_accuracy = mean_accuracy - zero_shot_per_model[model_id]
            std = data[model_id]["std"]
            lines[model_id]["mean_line"]["x"].append(k)
            lines[model_id]["mean_line"]["y"].append(mean_accuracy)
            lines[model_id]["mean_line"]["std"].append(std)
            results_per_line = data[model_id]["template_results"]
            for i, template in enumerate(results_per_line.keys()):
                if i + 1 not in lines[model_id]:
                    lines[model_id][i + 1] = {"x": [], "y": []}
                lines[model_id][i + 1]["x"].append(k)
                y_result = results_per_line[template]
                if k == 0:
                    zero_shot_per_model_per_template[model_id][template] = y_result
                if not use_differences:
                    lines[model_id][i + 1]["y"].append(results_per_line[template])
                else:
                    lines[model_id][i + 1]["y"].append(
                        results_per_line[template] - zero_shot_per_model_per_template[model_id][template])

    model_classes = list(set(model_classes))

    best_per_model_class = {}
    for model_class in model_classes:
        if model_class in models_to_show:
            model_ids_class = ids_per_class[model_class]["model_ids"]
            scores_per_class = ids_per_class[model_class]["scores"]
            max_score = max(scores_per_class)
            idx_max = scores_per_class.index(max_score)
            best_per_model_class[model_class] = model_ids_class[idx_max]

    # add group lines
    colors = plt.cm.Dark2(np.linspace(0, 1, len(unique_groups)))
    group_colors = {}
    mean_per_group_per_k = {group: defaultdict(list) for group in unique_groups}
    for model_id, line in lines.items():
        for x, y in zip(line["mean_line"]["x"], line["mean_line"]["y"]):
            name = line["name"]
            if name in models_to_show and model_id == best_per_model_class[name]:
                mean_per_group_per_k[model_to_group[name]][x].append(y)
    for i, group in enumerate(unique_groups):
        group_colors[group] = colors[i]
        for k in k_shot:
            group_mean = np.mean(mean_per_group_per_k[group][k])
            group_std = np.std(mean_per_group_per_k[group][k])
            lines[group]["mean_line"]["x"].append(k)
            lines[group]["mean_line"]["y"].append(group_mean)
            lines[group]["mean_line"]["std"].append(group_std)
            lines[group]["name"] = group
            lines[group]["size"] = "None"

    linewidth = 3
    markersize = 10
    # Mean accuracy plot per model of accuracy v. k
    legend_d = {}
    legend_lines, legend_labels = [], []
    plt.figure(figsize=(16, 14), dpi=200)
    for i, model_class in enumerate(list(model_classes)):
        if model_class not in models_to_show:
            continue
        color = MODEL_COLORS[model_class]
        model_id = best_per_model_class[model_class]
        name = lines[model_id]["name"]
        size = lines[model_id]["size"]
        if name == "InstructGPT":
            zorder = 10
        else:
            zorder = None
        if name == "Cohere":
            zorder = 15
        line = plt.errorbar(lines[model_id]["mean_line"]["x"], lines[model_id]["mean_line"]["y"],
                            yerr=lines[model_id]["mean_line"]["std"], color=color, zorder=zorder,
                            label=f"{lines[model_id]['name']}-{size}", marker="o", linestyle=LINESTYLE_GROUPING[name],
                            markersize=markersize, linewidth=linewidth)
        if "<engine>" in name:
            label = name.replace("<engine>", size)
            label += "-unknown"
        else:
            label = f"{name}-{size}"
        legend_lines.append(line)
        legend_labels.append(label)
        if name not in legend_d:
            legend_d[f"{name}"] = {}
        legend_d[f"{name}"]["line"] = line
        legend_d[f"{name}"]["label"] = label
    if not normalize_metric and not use_differences:
        randomline = plt.hlines(y=50.0, xmin=k_shot[0], xmax=k_shot[-1], label="Random chance", linestyles="solid", color="red",
                                linewidth=linewidth)
    if not normalize_metric:
        legend_lines.append(randomline)
        legend_labels.append(f"Random chance")
        legend_d["Random chance"] = {}
        legend_d["Random chance"]["line"] = randomline
        legend_d["Random chance"]["label"] = f"Random chance"
    ordered_legend_lines = [legend_d[line_n]["line"] for line_n in label_order]
    ordered_legend_labels = [legend_d[line_n]["label"] for line_n in label_order]
    plt.legend(ordered_legend_lines, ordered_legend_labels, fontsize=23, loc="lower center", ncol=3)
    plt.xticks(k_shot, k_shot, fontsize=28)
    plt.yticks(fontsize=28)  # Set text labels.
    ymin = 0.
    ymax = 100.
    if use_differences:
        ymin = -5
        ymax = 17
    plt.ylim(bottom=ymin, top=ymax)
    plt.xlabel("In-context examples (k)", fontsize=32)
    plt.ylabel("Accuracy (%)", fontsize=32)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, top=0.9, right=0.9, left=0.1)
    if not use_differences:
        plt.title(f"The accuracy versus number of in-context examples (k)\nfor the best of each model class", fontsize=32)
    else:
        plt.title(f"Relative accuracy (w.r.t 0-shot) due to in-context examples.", fontsize=32)
    plt.savefig(os.path.join(results_folder, f"accuracy_v_k_all.png"))

    # Grouped per model class of accuracy v. k
    linewidth = 2
    markersize = 10
    marker_style_per_group = {
        "Example IT": "+",
        "Benchmark IT": "x",
        "Dialogue FT": "v",
        "Base": "o"
    }
    legend_d = {}
    legend_lines, legend_labels = [], []
    # plt.figure(figsize=(16, 14), dpi=200)
    fig = plt.figure(figsize=(24, 14), dpi=200)
    ax = fig.add_subplot(111)
    k_to_str = {0: "Zero",
                1: "One",
                5: "Five",
                10: "Ten",
                15: "Fifteen",
                30: "Thirty"}
    width = 0.15  # the width of the bars
    multiplier = -1.5
    ordered_groups = ["Example IT", "Base", "Benchmark IT", "Dialogue FT"]
    multipliers = [-1.5, -0.5, 0.5, 1.5]
    for i, model_class in enumerate(list(model_classes)):
        if model_class not in models_to_show:
            continue
        color = MODEL_COLORS[model_class]
        model_id = best_per_model_class[model_class]
        name = lines[model_id]["name"]
        size = lines[model_id]["size"]
        group = model_to_group[name]
        group_idx = ordered_groups.index(group)
        if name == "InstructGPT":
            zorder = 10
        else:
            zorder = None
        if name == "Cohere":
            zorder = 15
        x = [f"{k_to_str[k]}" for k in lines[model_id]["mean_line"]["x"]]
        x = np.arange(len(k_shot))
        multiplier = multipliers[group_idx]
        offset = width * multiplier
        line = ax.scatter(x + offset,
                          lines[model_id]["mean_line"]["y"],
                          # s=np.array(lines[model_id]["mean_line"]["std"]) * 100,
                          s=100,
                          c=color, zorder=zorder,
                          label=f"{lines[model_id]['name']}-{size}", marker=marker_style_per_group[group],
                          # linestyle=LINESTYLE_GROUPING[name],
                          # markersize=markersize,
                          # linewidth=linewidth
                          )
        if "<engine>" in name:
            label = name.replace("<engine>", size)
            label += "-unknown"
        else:
            label = f"{name}-{size}"
        legend_lines.append(line)
        legend_labels.append(label)
        if name not in legend_d:
            legend_d[f"{name}"] = {}
        legend_d[f"{name}"]["line"] = line
        legend_d[f"{name}"]["label"] = f"{label}".replace("unknown", "*")

    width = 0.15  # the width of the bars
    multiplier = -1.5
    x = np.arange(len(k_shot))
    all_rects = {}
    for i, group in enumerate(ordered_groups):
        if group not in groups_per_model:
            continue
        color = group_colors[group]
        offset = width * multiplier
        # x = np.array(lines[group]["mean_line"]["x"])
        y = lines[group]["mean_line"]["y"]
        rects = ax.bar(x + offset, y,
                       width,
                       label=group, color=color, alpha=0.55)
        # ax.bar_label(rects, padding=3)
        multiplier += 1
        all_rects[group] = rects

    if not normalize_metric and not use_differences:
        randomline = ax.hlines(y=50.0, xmin=x[0], xmax=x[-1], label="Random chance", linestyles="solid",
                               color="red",
                               linewidth=linewidth)
    if not normalize_metric:
        legend_lines.append(randomline)
        legend_labels.append(f"Random chance")
        legend_d["Random chance"] = {}
        legend_d["Random chance"]["line"] = randomline
        legend_d["Random chance"]["label"] = f"Random chance"
    ordered_legend_lines = [legend_d[line_n]["line"] for line_n in label_order]
    ordered_legend_labels = [legend_d[line_n]["label"] for line_n in label_order]
    ordered_rects = [all_rects[label] for label in ordered_groups if label in groups_per_model]
    ordered_legend_lines = ordered_legend_lines
    ordered_legend_labels = ordered_legend_labels
    legend_one = plt.legend(ordered_legend_lines, ordered_legend_labels, fontsize=35, loc='center left',
                            bbox_to_anchor=(1, 0.4), ncol=1)
    ordered_groups_filtered = [group for group in ordered_groups if group in groups_per_model]
    ax = plt.gca().add_artist(legend_one)
    plt.legend(ordered_rects, ordered_groups, fontsize=36, loc="upper center", ncol=4)
    plt.xticks(np.arange(len(k_shot)), [k_to_str[k] for k in k_shot], fontsize=30)
    plt.yticks(fontsize=30)  # Set text labels.
    ymin = 0.
    ymax = 100.
    if use_differences:
        ymin = -5
        ymax = 17
    plt.ylim(bottom=ymin, top=ymax)
    plt.xlabel("In-context examples (k)", fontsize=40)
    plt.ylabel("Accuracy (%)", fontsize=40)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1, top=0.9, right=0.7, left=0.2)
    if not use_differences:
        plt.title(f"The accuracy versus number of in-context examples (k)\nwith bars showing group means",
                  fontsize=40)
    else:
        plt.title(f"Relative accuracy (w.r.t 0-shot) due to in-context examples.", fontsize=32)
    plt.savefig(os.path.join(results_folder, f"accuracy_v_k_all_scatter.png"))


def get_human_performance_all_examples(file_path: str):
    with open(file_path, "r") as infile:
        csv_reader = csv.DictReader(infile)
        annotators = defaultdict(list)
        ground_truth = []
        examples = []
        for row in csv_reader:
            for key, value in row.items():
                if "annotator" in key.lower():
                    annotators[key].append(value.strip().lower())
                if "ground truth" in key.lower():
                    ground_truth.append(value.strip().lower())
            example = {
                "source": "",
                "type": row["Ground truth"].strip().lower(),
                "utterance": row["Utterance"],
                "response": row["Response"],
                "implicature": row["Ground truth"].strip().lower(),
            }
            examples.append(example)
    accuracies = []
    all_correct = [0] * 150
    for key, choices in annotators.items():
        correct = np.array(choices) == np.array(ground_truth)
        all_correct += correct
        accuracy = sum(correct) / len(correct) * 100.0
        assert len(correct) == 150, "Wrong number of annotations."
        accuracies.append(accuracy)
    return accuracies, examples, all_correct


def get_all_human_performance(human_eval_files):
    all_correct_counts = []
    all_examples = []
    for human_eval_file in human_eval_files:
        accuracies, examples, all_correct = get_human_performance_all_examples(
            human_eval_file
        )
        all_correct_counts.extend(all_correct)
        all_examples.extend(examples)
    return all_examples, all_correct_counts


def extract_counter(results, k_shot):
    # k_shot = [0]
    # Extract and group the accuracies per prompt template.
    # Get the average accuracy overall, grouping together different models and k
    template_scores_overall = defaultdict(list)
    # Get the accuracy per k-shot, grouping together different models
    template_scores_per_k = defaultdict(lambda: defaultdict(list))
    # Get the accuracy per model, grouping together k
    template_scores_per_model = defaultdict(lambda: defaultdict(list))
    # Get the accuracy per model per k, not grouping anything
    template_scores_per_model_per_k = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    example_results = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    models_sorted_by_size = defaultdict(list)
    model_size_ids_sorted = defaultdict(list)
    model_sizes = defaultdict(list)
    model_size_ids = defaultdict(list)
    model_to_group = {}
    for k in k_shot:
        for model, model_results in results[str(k)].items():
            if "nightly" in model:
                continue
            if k == 0:
                model_size, size_id, name = parse_model_id(model)
                models_sorted_by_size[name].append(model)
                model_size_ids_sorted[name].append(size_id)
                model_sizes[name].append(model_size)
                model_size_ids[name].append(size_id)
                model_to_group[model] = name
            accuracy_per_template = model_results["template_results"]
            # template_rating = model_results["template_rating"]
            # likelihood_diff_per_template = model_results["likelihood_diffs_per_template"]
            for template in accuracy_per_template.keys():
                # Accuracy metrics
                template_scores_overall[template].append(
                    accuracy_per_template[template]
                )
                template_scores_per_k[k][template].append(
                    accuracy_per_template[template]
                )
                template_scores_per_model[model][template].append(
                    accuracy_per_template[template]
                )
                template_scores_per_model_per_k[model][k][template].append(
                    accuracy_per_template[template]
                )
                if "example_results" in model_results:
                    example_results[model][k][template] = model_results["example_results"][
                        template
                    ]
        if k == 0:
            for model in models_sorted_by_size:
                models_sorted_by_size[model] = [
                    x
                    for _, x in sorted(
                        zip(model_sizes[model], models_sorted_by_size[model])
                    )
                ]
                model_size_ids_sorted[model] = [
                    x
                    for _, x in sorted(
                        zip(model_sizes[model], model_size_ids_sorted[model])
                    )
                ]
                model_sizes[model] = sorted(model_sizes[model])
    examples_correct_count = Counter()
    example_data = {}
    counter_per_model_per_k = defaultdict(lambda: defaultdict(Counter))
    counter_per_model_per_k_shot_per_template = defaultdict(lambda: defaultdict(lambda: defaultdict(Counter)))
    for model, examples_per_model in example_results.items():
        for k, examples_per_model_per_k in examples_per_model.items():
            for template, examples in examples_per_model_per_k.items():
                for example_id, ex_results in examples.items():
                    example_id = str(int(example_id) % 600)
                    if example_id not in example_data:
                        cleaned_example = {}
                        for key, value in ex_results["original_example"].items():
                            cleaned_example[key] = value.replace("\r", "")
                        example_data[example_id] = cleaned_example
                    else:
                        cleaned_example = {}
                        for key, value in ex_results["original_example"].items():
                            cleaned_example[key] = value.replace("\r", "")
                        assert example_data[example_id] == cleaned_example
                    example_correct = ex_results["correct"]
                    if example_correct:
                        examples_correct_count[example_id] += 1
                        counter_per_model_per_k[model][k][example_id] += 1
                        counter_per_model_per_k_shot_per_template[model][k][template][example_id] += 1
                    else:
                        counter_per_model_per_k[model][k][example_id] += 0
                        counter_per_model_per_k_shot_per_template[model][k][template][example_id] += 0
    return counter_per_model_per_k, counter_per_model_per_k_shot_per_template, example_data, models_sorted_by_size, model_sizes, model_size_ids_sorted


def type_label_analysis(project_folder: str, models_to_show=["Cohere", "text-<engine>-001"],
                        k_shot=[0, 1, 5, 10, 15, 30]):
    # assert len(models_to_show) == 2, "Only implemented for 2 models to show"
    file = "data/type_labels.csv"
    with open(file, "r") as infile:
        file_reader = csv.reader(infile)
        keys = []
        examples_with_type_labels = {}
        for i, row in enumerate(file_reader):
            if i == 0:
                keys.extend(row)
            else:
                example = {
                    "utterance": row[0],
                    "response": row[1],
                    "implicature": row[2].strip(".").lower(),
                    "label": row[3],
                    "factual": row[4],
                }
                examples_with_type_labels[row[0].replace("\r", "")] = example

    results_file = os.path.join(project_folder, "all_results.json")
    with open(results_file, "r") as infile:
        results = json.load(infile)

    human_eval_files = [
        "data/human_eval/human_eval - 1-150.csv",
        "data/human_eval/human_eval - 151-300.csv",
        "data/human_eval/human_eval - 301-450.csv",
        "data/human_eval/human_eval - 451-600.csv",
    ]
    human_examples, human_counts = get_all_human_performance(human_eval_files)

    (
        counter_per_model_per_k,
        counter_per_model_per_k_shot_per_template,
        example_data,
        models_sorted_by_size,
        model_sizes,
        size_ids_sorted_by_size
    ) = extract_counter(results, k_shot)

    # Get the example IDs for the labeled examples
    examples_with_type_labels_id = {}
    found_labels = 0
    label_dist = Counter()

    for ex_id, example in example_data.items():
        key = example["utterance"].replace("\r", "")
        if key in examples_with_type_labels:
            example_labeled = examples_with_type_labels[
                example["utterance"].replace("\r", "")
            ]
            found_labels += 1
        else:
            example_labeled = {"label": "Other"}
            example_labeled.update(example)
        label_dist[example_labeled["label"]] += 1
        label_dist["Mean"] += 1
        examples_with_type_labels_id[ex_id] = example_labeled

    # Let's select a view models to show this for, change below to view different models
    print(f"Options of model groups to show: {list(models_sorted_by_size.keys())}")
    # models_to_show
    # models_to_show = ["chatgpt"]
    line_style_group = {models_to_show[0]: "solid", models_to_show[1]: "dashed"}
    for model in models_to_show:
        print(f"Will show {model}")
        print(f"-- with sizes: {models_sorted_by_size[model]}")

    labels = list(label_dist.keys())
    look_at_labels = [
        "World knowledge",
        "Idiom",
        "Rhetorical question",
        "Particularised",
        "Generalised",
        "Other",
    ]
    num_labels = len(look_at_labels)

    lines = {}
    results_table = {"Model": [], "Mean": []}
    for label_type in look_at_labels:
        results_table[label_type] = []
    tables_per_k = {k: copy.deepcopy(results_table) for k in k_shot}
    all_type_label_results_d = {}

    for group_name in models_to_show:
        lines[group_name] = {}
        all_type_label_results_d[group_name] = {}
        for k in k_shot:
            all_type_label_results_d[group_name][k] = {}
            lines[group_name][k] = {
                label: {"x": [], "y": [], "y_absolute": [], "std": []} for label in look_at_labels
            }
            lines[group_name][k]["Mean"] = {"x": [], "y": [], "y_absolute": [], "std": []}
            print(f"---------------- {k}-shot")
            # lines[group_name]["Mean"] = {}
            # = {"x": [], "y": [], "std": []}
            sorted_models = models_sorted_by_size[group_name]
            sorted_sizes = model_sizes[group_name]
            sorted_size_ids = size_ids_sorted_by_size[group_name]
            print(f"- Model group {group_name}")
            for i, model in enumerate(sorted_models):
                print(f"----- Model {model}")
                if "<engine>" in group_name:
                    size_id = "unknown"
                    table_group_name = group_name.replace("<engine>", sorted_size_ids[i])
                else:
                    size_id = sorted_size_ids[i]
                    table_group_name = group_name
                all_type_label_results_d[group_name][k][size_id] = {}
                tables_per_k[k]["Model"].append(f"{table_group_name}-{size_id}")
                correct_per_type = Counter()
                correct_per_k_per_template_per_type = defaultdict(lambda: defaultdict(Counter))
                for ex_id, example_labeled in examples_with_type_labels_id.items():
                    example_correct_count_zero_shot = counter_per_model_per_k[model][k][
                        ex_id
                    ]
                    correct_per_type[example_labeled["label"]] += (
                        example_correct_count_zero_shot / 6
                    )
                    for temp_idx in range(6):
                        example_correct_count_k_shot_template = counter_per_model_per_k_shot_per_template[model][k][f"prompt_template_{temp_idx + 1}"][ex_id]
                        correct_per_k_per_template_per_type[k][temp_idx][example_labeled["label"]] += example_correct_count_k_shot_template
                        correct_per_k_per_template_per_type[k][temp_idx]["Mean"] += example_correct_count_k_shot_template
                    correct_per_type["Mean"] += example_correct_count_zero_shot / 6
                mean_correct = correct_per_type["Mean"]
                percentage_correct_mean = mean_correct / label_dist["Mean"] * 100
                for j, label_type in enumerate(look_at_labels):
                    type_correct = correct_per_type[label_type]
                    perc_per_template = []
                    for temp_idx in range(6):
                        type_correct_per_template = correct_per_k_per_template_per_type[k][temp_idx][label_type]
                        percentage_correct_template_label = (
                            type_correct_per_template / label_dist[label_type] * 100
                        )
                        perc_per_template.append(percentage_correct_template_label)
                    percentage_correct_label = (
                        type_correct / label_dist[label_type] * 100
                    )
                    std_correct_label = np.std(perc_per_template)
                    lines[group_name][k][label_type]["y"].append(
                        percentage_correct_label - percentage_correct_mean
                    )
                    lines[group_name][k][label_type]["y_absolute"].append(
                        percentage_correct_label
                    )
                    lines[group_name][k][label_type]["x"].append(sorted_sizes[i])
                    lines[group_name][k][label_type]["std"].append(std_correct_label)
                    lines[group_name][k]["Mean"]["x"].append(sorted_sizes[i])
                    lines[group_name][k]["Mean"]["y_absolute"].append(percentage_correct_mean)
                    lines[group_name][k]["Mean"]["std"].append(0.)
                    tables_per_k[k][label_type].append(f"{percentage_correct_label:.2f} +/- {std_correct_label:.2f}")
                    all_type_label_results_d[group_name][k][size_id][label_type] = {
                        "mean": percentage_correct_label,
                        "std": std_correct_label
                    }
                    print(
                        f"{label_type} absolute label mean: {percentage_correct_label}"
                    )
                    print(
                        f"{label_type} absolute label std: {std_correct_label}"
                    )
                    print(f"{label_type} absolute mean: {percentage_correct_mean}")
                tables_per_k[k]["Mean"].append(f"{percentage_correct_mean:.2f}")

    print(f"- Model group humans")
    correct_per_type_humans = Counter()
    for ex_id, example_labeled in examples_with_type_labels_id.items():
        assert (
            human_examples[int(ex_id)]["utterance"]
            .replace("\r", "")
            .replace("\n", "")
            .strip()
            == example_data[ex_id]["utterance"]
            .replace("\r", "")
            .replace("\n", "")
            .strip()
        )
        example_correct_count_zero_shot = human_counts[int(ex_id)]
        correct_per_type_humans[example_labeled["label"]] += (
            example_correct_count_zero_shot / 5
        )
        correct_per_type_humans["Mean"] += example_correct_count_zero_shot / 5
    mean_correct_humans = correct_per_type_humans["Mean"]
    percentage_correct_mean_humans = mean_correct_humans / label_dist["Mean"] * 100
    all_type_label_results_d["Humans"] = {}
    for k in k_shot:
        tables_per_k[k]["Model"].append("Humans")
        tables_per_k[k]["Mean"].append(f"{percentage_correct_mean_humans:.2f}")
        all_type_label_results_d["Humans"]["Mean"] = percentage_correct_mean_humans
    for j, label_type in enumerate(look_at_labels):
        type_correct = correct_per_type_humans[label_type]
        percentage_correct_label = type_correct / label_dist[label_type] * 100
        # lines[group_name][label_type]["y"].append(percentage_correct_label - percentage_correct_mean_humans)
        # lines[group_name][label_type]["x"].append(sorted_sizes[i])
        print(f"{label_type} absolute label mean: {percentage_correct_label}")
        print(f"{label_type} absolute mean: {percentage_correct_mean_humans}")
        all_type_label_results_d["Humans"][label_type] = percentage_correct_label
        for k in k_shot:
            tables_per_k[k][label_type].append(f"{percentage_correct_label:.2f}")

    with open(f"{project_folder}/all_type_label_results.json", "w") as outfile:
        json.dump(all_type_label_results_d, outfile, indent=4)

    labels_first_table = look_at_labels[:3]
    labels_second_table = look_at_labels[3:]
    for k in k_shot:
        first_table = {"Model": tables_per_k[k]["Model"], "Mean": tables_per_k[k]["Mean"], **{label: tables_per_k[k][label] for label in labels_first_table}}
        second_table = {"Model": tables_per_k[k]["Model"], "Mean": tables_per_k[k]["Mean"],
                       **{label: tables_per_k[k][label] for label in labels_second_table}}
        df_1 = pd.DataFrame(first_table)
        df_2 = pd.DataFrame(second_table)
        print(df_1.to_latex(index=False,
                          position="ht",
                          label=f"type-label-analysis-{k}-shot-1",
                          caption=f"Accuracy per label for model group HF FT for {k}-shot evaluation.",
                          bold_rows=True,
                          column_format="l" + "c" * (len(labels_first_table) + 1)))
        print(df_2.to_latex(index=False,
                            position="ht",
                            label=f"type-label-analysis-{k}-shot-2",
                            caption=f"Accuracy per label for model group HF FT for {k}-shot evaluation.",
                            bold_rows=True,
                            column_format="l" + "c" * (len(labels_second_table) + 1)))

    linewidth = 3
    markersize = 10

    # Relative plot
    for k in k_shot:
        colors = plt.cm.Dark2(np.linspace(0, 1, num_labels))
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(20, 10), dpi=200)
        legend_lines, legend_labels = [], []
        x_min = float("inf")
        x_max = 0
        saved_colors = {}
        for i, group_name in enumerate(models_to_show[:2]):
            # plt.figure(figsize=(16, 14), dpi=200)
            for j, (line_key, line_data) in enumerate(lines[group_name][k].items()):
                if line_key == "Mean":
                    # Don't plot mean line for relative plot (mean line is y = 0)
                    continue
                line, = ax[i].plot(
                    line_data["x"],
                    line_data["y"],
                    label=f"Prompt template {line_key}",
                    marker="o",
                    color=colors[j],
                    linestyle="solid",
                    markersize=markersize,
                    linewidth=linewidth,
                )
                saved_colors[line_key] = colors[j]
                if min(line_data["x"]) < x_min:
                    x_min = min(line_data["x"])
                if max(line_data["x"]) > x_max:
                    x_max = max(line_data["x"])
                if i == 0:
                    legend_lines.append(line)
                    legend_labels.append(f"{line_key}")
            plt.xscale("log")
            ax[i].xaxis.set_tick_params(labelsize=24)
            ax[i].yaxis.set_tick_params(labelsize=24)
            ax[i].title.set_text(f"{group_name}")
            ax[i].title.set_size(24)
            if i == 0:
                ax[i].set_ylabel("Relative Accuracy (%)", fontsize=28)
            ax[i].set_xlabel("Model Parameters (Non-Embedding)", fontsize=28)
        for i in range(len(ax)):
            randomline = ax[i].hlines(
                y=0.0,
                xmin=x_min,
                xmax=x_max,
                label="Mean accuracy",
                linestyles="dotted",
                color="black",
                linewidth=linewidth,
            )
        legend_lines.append(randomline)
        legend_labels.append("All types")
        ax[0].legend(legend_lines, legend_labels, fontsize=20, loc="lower left")
        plt.suptitle(
            f"Relative accuracy (w.r.t. mean) for each type of implicature.", fontsize=28
        )
        plt.subplots_adjust(top=0.85)
        plt.tight_layout()
        plt.savefig(f"{project_folder}/type_labels_plot_{k}-shot.jpg")

    for k in k_shot:
        # Absolute plot
        saved_colors["Mean"] = "black"
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(20, 10), dpi=200)
        legend_lines, legend_labels = [], []
        x_min = float("inf")
        x_max = 0
        for i, group_name in enumerate(models_to_show[:2]):
            # plt.figure(figsize=(16, 14), dpi=200)
            for j, (line_key, line_data) in enumerate(lines[group_name][k].items()):
                if line_key not in ["Particularised", "Generalised", "Other", "Mean"]:
                    continue
                line = ax[i].errorbar(
                    line_data["x"],
                    line_data["y_absolute"],
                    yerr=line_data["std"],
                    label=f"Prompt template {line_key}",
                    marker="o",
                    color=saved_colors[line_key],
                    linestyle="solid" if line_key != "Mean" else "dotted",
                    markersize=markersize,
                    linewidth=linewidth,
                )
                if min(line_data["x"]) < x_min:
                    x_min = min(line_data["x"])
                if max(line_data["x"]) > x_max:
                    x_max = max(line_data["x"])
                if i == 0:
                    legend_lines.append(line)
                    if line_key == "Mean":
                        line_key = "All types"
                    legend_labels.append(f"{line_key}")
            plt.xscale("log")
            ax[i].xaxis.set_tick_params(labelsize=24)
            ax[i].yaxis.set_tick_params(labelsize=24)
            ax[i].title.set_text(f"{group_name}")
            ax[i].title.set_size(24)
            ax[i].set_ylim(0., 100.)
            if i == 0:
                ax[i].set_ylabel("Accuracy (%)", fontsize=28)
            ax[i].set_xlabel("Model Parameters (Non-Embedding)", fontsize=28)
        for i in range(len(ax)):
            randomline = ax[i].hlines(
                y=50.0,
                xmin=x_min,
                xmax=x_max,
                label="Random chance",
                linestyles="solid",
                color="red",
                linewidth=linewidth,
            )
        legend_lines.append(randomline)
        legend_labels.append("Random chance")
        ax[0].legend(legend_lines, legend_labels, fontsize=20, loc="lower left")
        plt.suptitle(
            f"Accuracy for each type of implicature.", fontsize=28
        )
        plt.subplots_adjust(top=0.85)
        plt.tight_layout()
        plt.savefig(f"{project_folder}/type_labels_plot_absolute_{k}-shot.jpg")

    print(f"Found {found_labels} non-other labeled examples")
    print(f"Distributed as: {label_dist}")
    print(f"Total count: {sum(label_dist.values()) - label_dist['Mean']}")


def write_to_csv(all_results_file: str):
    models_to_write = ["openai-davinci", "openai-text-davinci-001", "openai-text-davinci-002"]
    with open(all_results_file, "r") as infile:
        all_results = json.load(infile)

    num_examples = 600
    lines = [["model", "k", "ID", "utterance", "response", "implicature", "1", "2", "3", "4", "5", "6"]]
    for k in all_results.keys():
        for model in models_to_write:
            model_k_results = all_results[k][model]
            for ex_id in range(num_examples):
                utterance = model_k_results["example_results"][f"prompt_template_1"][str(ex_id)]["original_example"]["utterance"]
                response = model_k_results["example_results"][f"prompt_template_1"][str(ex_id)]["original_example"]["response"]
                true = model_k_results["example_results"][f"prompt_template_1"][str(ex_id)]["original_example"]["implicature"]
                line = [model, k, ex_id, utterance, response, true]
                for prompt_idx in range(1, 7):
                    example_result = model_k_results["example_results"][f"prompt_template_{prompt_idx}"][str(ex_id)]
                    line.append(example_result["pred"] == true)
                lines.append(line)
    with open("error_analysis/example_results_openai.csv", "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerows(lines)
    return


def prepare_chatgpt_examples(example_batch, add_answers=False, use_template=False):
    if not add_answers:
        num_columns = 2
    else:
        num_columns = 3
    quoted_examples = []
    for example in example_batch:
        example[2] = example[2][:4].strip()
        if not use_template:
            quoted_example = []
            for col in example:
                quoted_example.append(f'\"{col}\"')
            quoted_examples.append(','.join(quoted_example[:num_columns]) + "\n")
        else:
            if add_answers:
                implicature = example[2].replace(".", "").replace('\"', "").lower()
                quoted_example = f'Esther asked \"{example[0]}\" and Juan responded \"{example[1]}\", which means {implicature}.\n"'
            else:
                quoted_example = f'Esther asked \"{example[0]}\" and Juan responded \"{example[1]}\", which means\n'
            quoted_examples.append(quoted_example)
    example_string = f"{''.join(quoted_examples)}"
    return example_string


def make_chatgpt_string(prompt, column_names, examples, num_examples_per_prompt, k_shot, dev_examples, folder_path,
                        file_name, shuffle, use_template):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    if k_shot > 0:
        prompt += f" For the first {k_shot} lines the correct value is already provided."
        if not use_template:
            prompt += f"\n\n{','.join(column_names[:2])}\n"
        else:
            prompt += "\n\n"
        k_shot_examples = dev_examples[:k_shot]
        k_shot_string = prepare_chatgpt_examples(k_shot_examples, add_answers=True, use_template=use_template)
        prompt += k_shot_string
    else:
        if not use_template:
            prompt += f"\n\n{','.join(column_names[:2])}\n"
        else:
            prompt += "\n\n"

    if shuffle:
        random.shuffle(examples)
    for i in range(0, len(examples), num_examples_per_prompt):
        example_batch = examples[i:i + num_examples_per_prompt]
        example_string = prepare_chatgpt_examples(example_batch, add_answers=False, use_template=use_template)

        with open(
                f"{folder_path}/{file_name}_{i}-{i + num_examples_per_prompt - 1}.txt",
                "w") as outfile:
            outfile.write(f"{prompt}{example_string}")


def read_csv_file(file):
    with open(file, "r") as infile:
        csv_reader = csv.reader(infile)
        data = []
        for row in csv_reader:
            data.append(row)
        column_names = data[0]
        examples = data[1:]
    return column_names, examples


def generate_chatgpt_eval_files():
    test_csv_file = "data/test_conversational_implicatures.csv"
    column_names, examples = read_csv_file(test_csv_file)
    dev_csv_file = "data/dev_conversational_implicatures.csv"
    dev_column_names, dev_examples = read_csv_file(dev_csv_file)
    random.shuffle(dev_examples)

    prompt = f"Below is a csv file of a Q and A dataset. The columns are Context utterance and Response utterance, " \
             "The context utterance column contains questions asked by Speaker A. The response utterance column " \
             "contains the answer provided by Speaker B in response. For each line of the csv, repeat the line and add the correct " \
             "value of yes or no, which specifies whether Speaker B' response (in the Response utterance column) in " \
             f"response to Speaker A's question (in the Context utterance column) means yes or no."

    templated_prompt = f"Below is a list of sentences. In every line, Esther asks Juan a question, and Juan responds. " \
                        "Each response implies either yes or no. For each line of the csv, repeat the sentence and add the correct " \
                        "value of yes or no, which specifies whether Juan's response in " \
                        f"response to Esther's question means yes or no."

    num_examples_per_prompt = 25
    k_shot = 0

    # No shuffle 0-shot
    folder_path = "data/wrong_chatGPT/no_template_no_shuffle_zero_shot"
    file_name = "no_template_no_shuffle_zero_shot"
    make_chatgpt_string(prompt, column_names, examples, num_examples_per_prompt, k_shot, dev_examples, folder_path, file_name,
                        shuffle=False, use_template=False)

    # No shuffle template 0-shot
    folder_path = "data/wrong_chatGPT/template_no_shuffle_zero_shot"
    file_name = "template_no_shuffle_zero_shot"
    make_chatgpt_string(templated_prompt, column_names, examples, num_examples_per_prompt, k_shot, dev_examples, folder_path,
                        file_name,
                        shuffle=False, use_template=True)

    # No shuffle 5-shot
    k_shot = 5
    num_examples_per_prompt = 20
    folder_path = "data/chatGPT/no_template_no_shuffle_five_shot"
    file_name = "no_template_no_shuffle_five_shot"
    make_chatgpt_string(prompt, column_names, examples, num_examples_per_prompt, k_shot, dev_examples, folder_path, file_name,
                        shuffle=False, use_template=False)

    # No shuffle template 5-shot
    k_shot = 5
    num_examples_per_prompt = 20
    folder_path = "data/chatGPT/template_no_shuffle_five_shot"
    file_name = "template_no_shuffle_five_shot"
    make_chatgpt_string(templated_prompt, column_names, examples, num_examples_per_prompt, k_shot, dev_examples,
                        folder_path,
                        file_name,
                        shuffle=False, use_template=True)

    # Shuffle one 0-shot
    k_shot = 0
    num_examples_per_prompt = 25
    folder_path = "data/wrong_chatGPT/no_template_shuffle_one_zero_shot"
    file_name = "no_template_shuffle_one_zero_shot"
    make_chatgpt_string(prompt, column_names, examples, num_examples_per_prompt, k_shot, dev_examples, folder_path, file_name,
                        shuffle=True, use_template=False)

    # Shuffle one template 0-shot
    k_shot = 0
    num_examples_per_prompt = 25
    folder_path = "data/wrong_chatGPT/template_shuffle_one_zero_shot"
    file_name = "template_shuffle_one_zero_shot"
    make_chatgpt_string(templated_prompt, column_names, examples, num_examples_per_prompt, k_shot, dev_examples,
                        folder_path,
                        file_name,
                        shuffle=False, use_template=True)

    # Shuffle one 5-shot
    k_shot = 5
    num_examples_per_prompt = 20
    folder_path = "data/chatGPT/no_template_shuffle_one_five_shot"
    file_name = "no_template_shuffle_one_five_shot"
    make_chatgpt_string(prompt, column_names, examples, num_examples_per_prompt, k_shot, dev_examples, folder_path, file_name,
                        shuffle=False, use_template=False)

    # Shuffle one template 5-shot
    k_shot = 5
    num_examples_per_prompt = 20
    folder_path = "data/chatGPT/template_shuffle_one_five_shot"
    file_name = "template_shuffle_one_five_shot"
    make_chatgpt_string(templated_prompt, column_names, examples, num_examples_per_prompt, k_shot, dev_examples,
                        folder_path,
                        file_name,
                        shuffle=False, use_template=True)


def find_example_idx(examples, found_example):
    for i, test_example in enumerate(examples):
        found_parts = 0
        for part in test_example[:-1]:
            if part.lower().strip("\n") in found_example.lower().strip("\n"):
                found_parts += 1
        if found_parts == 2:
            return i


def check_example_same(dataset_example, found_example):
    for part in dataset_example[:-1]:
        assert part.lower().replace("\n", "").replace(" ", "").strip("\n") in found_example.lower().replace("\n", "").replace(" ", "").strip("\n")


def get_type_labels(examples):
    file = "data/type_labels.csv"
    with open(file, "r") as infile:
        file_reader = csv.reader(infile)
        keys = []
        examples_with_type_labels = {}
        for i, row in enumerate(file_reader):
            if i == 0:
                keys.extend(row)
            else:
                example = {
                    "utterance": row[0],
                    "response": row[1],
                    "implicature": row[2].strip(".").lower(),
                    "label": row[3],
                    "factual": row[4],
                }
                examples_with_type_labels[row[0].replace("\r", "").strip()] = example

    # Get the example IDs for the labeled examples
    examples_with_type_labels_id = {}
    found_labels = 0
    label_dist = Counter()

    for ex_id, example in enumerate(examples):
        key = example[0].replace("\r", "").strip()
        if key in examples_with_type_labels:
            example_labeled = examples_with_type_labels[
                example[0].replace("\r", "").strip()
            ]
            found_labels += 1
        else:
            example_labeled = {"label": "Other", "example": example}
        label_dist[example_labeled["label"]] += 1
        label_dist["Mean"] += 1
        examples_with_type_labels_id[ex_id] = example_labeled
    return examples_with_type_labels_id


def eval_chatgpt_files(folder, shuffled, templated):
    test_csv_file = "data/test_conversational_implicatures.csv"
    column_names, test_examples = read_csv_file(test_csv_file)
    dev_csv_file = "data/dev_conversational_implicatures.csv"
    dev_column_names, dev_examples = read_csv_file(dev_csv_file)

    examples = []
    test_examples_ordered = []
    correct_answers = []
    chatgpt_answers = []
    full_chatgpt_answers = []
    test_example_ids = []
    type_labels = []

    type_labeled_examples = get_type_labels(test_examples)

    file_counter = 0
    # Loop over the files in the folder
    for file in os.listdir(folder):

        # If the file is answered, extract answers
        if file.endswith("answered.txt"):

            # Filename contains number of examples in this file
            file_example_indices = file.split("_")[-2].split("-")
            first_example_idx = int(file_example_indices[0])
            num_examples = int(file_example_indices[1]) - int(file_example_indices[0]) + 1

            # Loop over lines and extract the examples and the answers
            with open(os.path.join(folder, file), "r") as infile:

                # Keep track of whether we are still at prompt or already at answers by chatGPT
                at_answers = False
                current_line = ""
                current_answer = ""
                current_example_idx = first_example_idx

                file_examples = []
                file_chatgpt_answers = []
                file_correct_answers = []
                file_test_examples = []
                file_full_chatgpt_answers = []
                file_test_example_ids = []
                file_type_labels = []

                # Each line contains an example or an answered example
                for line in infile.readlines():

                    # Skip this line
                    if "Context utterance" in line or "Below is a list of sentences" in line:
                        continue

                    # if the line contains "ANSWER:" we are at the answers
                    if "ANSWER:" in line:
                        at_answers = True
                        continue

                    # If not yet at the answers, save the example
                    if not at_answers:

                        # Examples can contain linebreaks, so add to current example until we have 4 "'s
                        if not templated and current_line.count('"') < 4:
                            current_line += line
                        elif templated and ("which means" not in current_line.lower()):
                            current_line += line
                        # Current example is done, add to examples and increment pointer
                        else:
                            # Skip line if it's a dev example
                            dev_example_idx = find_example_idx(dev_examples, current_line.strip("\n"))
                            if dev_example_idx is not None:
                                current_line = line
                                continue
                            file_examples.append(current_line.strip("\n"))

                            # Match the example to the test dataset index
                            if not shuffled:
                                example_idx = current_example_idx
                            else:
                                example_idx = find_example_idx(test_examples, file_examples[-1])

                            check_example_same(test_examples[example_idx], file_examples[-1])
                            file_test_examples.append(test_examples[example_idx])
                            correct_answer = test_examples[example_idx][-1].lower().strip(".")
                            correct_answer = correct_answer.split(" ")[0].strip().lower().strip(".").strip(",").strip('"')
                            file_correct_answers.append(correct_answer)
                            file_test_example_ids.append(example_idx)
                            file_type_labels.append(type_labeled_examples[example_idx]["label"])

                            current_example_idx += 1
                            current_line = line

                    # We are at the answers, only extract yes or no.
                    else:
                        # Examples can contain linebreaks, so add to current example until we have 4 "'s
                        # Examples can contain linebreaks, so add to current example until we have 4 "'s
                        if not templated and current_answer.count('"') < 4:
                            current_answer += line
                        elif templated and ("which means" not in current_answer.lower()):
                            current_answer += line
                        # Current example is done, add to examples and increment pointer
                        else:
                            # answer = line.split(",")[-1].strip().lower().strip(".").strip('"')
                            # Skip line if it's a dev example
                            dev_example_idx = find_example_idx(dev_examples, current_answer)
                            if dev_example_idx is not None:
                                current_answer = line
                                continue
                            answer = current_answer.split(",")[-1].lower().replace("\n", "").replace(".", "").replace('"', "")[-3:].strip()
                            if not answer:
                                continue
                            file_full_chatgpt_answers.append(current_answer)
                            assert answer in ["yes", "no"], f"Answer is {answer}"
                            file_chatgpt_answers.append(answer)
                            current_answer = line
                # Process last answer as well
                answer = current_answer.split(",")[-1].lower().replace("\n", "").replace(".", "").replace('"', "")[-3:].strip()
                if answer:
                    assert answer in ["yes", "no"], f"Answer is {answer}"
                    file_chatgpt_answers.append(answer)
                    file_full_chatgpt_answers.append(current_answer)
            assert len(file_examples) == num_examples
            assert len(file_correct_answers) == num_examples
            assert len(file_chatgpt_answers) == num_examples
            examples.extend(file_examples)
            correct_answers.extend(file_correct_answers)
            chatgpt_answers.extend(file_chatgpt_answers)
            test_examples_ordered.extend(file_test_examples)
            test_example_ids.extend(file_test_example_ids)
            full_chatgpt_answers.extend(file_full_chatgpt_answers)
            type_labels.extend(file_type_labels)
            file_counter += 1
            print(file)
            print(f"File accuracy: {(np.array(file_chatgpt_answers) == np.array(file_correct_answers)).sum() / len(file_correct_answers)}")
    folder_accuracy = (np.array(chatgpt_answers) == np.array(correct_answers)).sum() / len(correct_answers)
    print(f"Evaluated folder {folder}")
    print(f"Num examples answered: {len(correct_answers)}")
    print(f"Folder accuracy: {folder_accuracy * 100}")
    deepest_folder = folder.split("/")[-1]
    labeled_accuracy = {
        "particularised": {"correct": 0, "total": 0},
        "generalised": {"correct": 0, "total": 0},
        "other": {"correct": 0, "total": 0},
    }
    with open(os.path.join(folder, f'{deepest_folder}.csv'), "w") as infile:
        writer = csv.writer(infile)
        writer.writerow(["Prompt example", "Example ID", "Context", "Response", "Implicature", "Type label",
                         "Full ChatGPT answer", "ChatGPT answer", "Correct answer"])
        for example, test_example_id, test_example, type_label, full_chatgpt_answer, chatgpt_answer, correct_answer in zip(
            examples, test_example_ids, test_examples_ordered, type_labels, full_chatgpt_answers, chatgpt_answers,
                correct_answers):
            current_row = [example, test_example_id] + test_example + [type_label, full_chatgpt_answer, chatgpt_answer,
                                                                       correct_answer]
            example_correct = chatgpt_answer == correct_answer
            if type_label.lower() in ["particularised", "generalised", "other"]:
                if example_correct:
                    labeled_accuracy[type_label.lower()]["correct"] += 1
                labeled_accuracy[type_label.lower()]["total"] += 1
            if type_label.lower() == "particularised":
                print(chatgpt_answer)
            writer.writerow(current_row)
    for type_label, type_results in labeled_accuracy.items():
        print("---- labelled accuracy ----")
        print(f"Accuracy label {type_label} is {(type_results['correct'] / type_results['total']) * 100}")
    print(f"Wrote results to {os.path.join(folder, f'{deepest_folder}.csv')}")


def find_example_idx_bb(examples, big_bench_example):
    regex = r"""
            [-,.;@#?!&$'’…"]+  # Accept one or more copies of punctuation
            \ *           # plus zero or more copies of a space,
            """
    for i, test_example in enumerate(examples):
        utterance_no_punct = re.sub(regex, ' ', test_example[0], flags=re.VERBOSE).lower().strip("\n").replace(" ", "")
        response_no_punct = re.sub(regex, ' ', test_example[1], flags=re.VERBOSE).lower().strip("\n").replace(" ", "")
        found_example_no_punct = re.sub(regex, ' ', big_bench_example, flags=re.VERBOSE)
        utterance_big_bench, response_big_bench = found_example_no_punct.split("Speaker 2:")
        utterance_big_bench_cleaned = utterance_big_bench.replace("Speaker 1: ", "").lower().strip("\n").replace(" ", "")
        response_big_bench_cleaned = response_big_bench.lower().strip("\n").replace(" ", "")
        utterance_edit_distance = editdistance.eval(utterance_no_punct, utterance_big_bench_cleaned)
        response_edit_distance = editdistance.eval(response_no_punct, response_big_bench_cleaned)
        if utterance_edit_distance + response_edit_distance < 5:
            return i


def compare_BIG_bench_task():

    # Get own data
    test_csv_file = "data/test_conversational_implicatures.csv"
    column_names, test_examples = read_csv_file(test_csv_file)
    dev_csv_file = "data/dev_conversational_implicatures.csv"
    dev_column_names, dev_examples = read_csv_file(dev_csv_file)

    type_labeled_examples = get_type_labels(test_examples)

    human_eval_files = [
        "data/human_eval/human_eval - 1-150.csv",
        "data/human_eval/human_eval - 151-300.csv",
        "data/human_eval/human_eval - 301-450.csv",
        "data/human_eval/human_eval - 451-600.csv",
    ]
    human_examples, human_counts = get_all_human_performance(human_eval_files)

    project_folder = "error_analysis"
    results_file = os.path.join(project_folder, "all_results.json")
    with open(results_file, "r") as infile:
        results = json.load(infile)
    (
        counter_per_model_per_k,
        counter_per_model_zero_shot_per_template,
        example_data,
        models_sorted_by_size,
        model_sizes,
        size_ids_sorted_by_size
    ) = extract_counter(results, k_shot=[0, 1, 5, 10, 15, 30])

    # Get BIG bench data
    file = "data/BIG_bench_task.json"
    with open(file, "r") as infile:
        big_bench_data = json.load(infile)

    human_performance_big_bench = 0
    davinci_performance_big_bench = 0
    big_bench_count = 0
    found_dev_examples = 0
    not_found_examples = 0
    example_idxs_found = []
    for example in big_bench_data["examples"]:
        dev_example_idx = find_example_idx_bb(dev_examples, example["input"])
        if dev_example_idx is not None:
            found_dev_examples += 1
            continue
        example_idx = find_example_idx_bb(test_examples, example["input"])
        if example_idx is None:
            print(f"{example['input']} not found")
            not_found_examples += 1
            continue
        example_idxs_found.append(example_idx)
        human_performance = (human_counts[example_idx] / 5) * 100.
        davinci_1_performance = (counter_per_model_per_k["openai-text-davinci-001"][0][f"{example_idx}"] / 6) * 100.
        matched_big_bench_example = {
            "example_idx": example_idx,
            "big_bench_input": example["input"],
            "original_example": test_examples[example_idx],
            "human_performance": human_performance,
            "davinci-001-performance": davinci_1_performance
        }
        human_performance_big_bench += human_performance
        davinci_performance_big_bench += davinci_1_performance
        big_bench_count += 1
    print(f"Found {big_bench_count} examples in test data.")
    print(f"Found {found_dev_examples} dev examples in BIG bench data.")
    print(f"Could not find {not_found_examples} examples.")
    print(f"Human performance: {human_performance_big_bench / big_bench_count}")
    print(f"text-davinci-001 performance: {davinci_performance_big_bench / big_bench_count}")

    ambiguous_human_performance = 0
    ambiguous_davinci_performance = 0
    ambiguous_count = 0
    for i in range(600):
        if i not in example_idxs_found:
            human_performance = (human_counts[i] / 5) * 100.
            davinci_1_performance = (counter_per_model_per_k["openai-text-davinci-001"][0][f"{i}"] / 6) * 100.
            ambiguous_human_performance += human_performance
            ambiguous_davinci_performance += davinci_1_performance
            ambiguous_count += 1

    print(f"Did not find {ambiguous_count} examples in BIG bench data.")
    print(f"Human performance on these: {ambiguous_human_performance / ambiguous_count}")
    print(f"Davinci performance on these: {ambiguous_davinci_performance / ambiguous_count}")


def print_latex_table(results_path):
    with open(results_path, "r") as infile:
        results = json.load(infile)
    results_folder = results_path.split("/")[0]
    k_shot = [int(k) for k in list(results.keys())]
    data_per_model = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    data_per_model_per_template = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    model_classes = defaultdict(int)
    for k in k_shot:
        data = results[str(k)]
        for model_id in data.keys():
            model_size, size_str, model_class = parse_model_id(model_id)
            print("------")
            print(model_class)
            print(model_size)
            print(size_str)
            print("------")
            if k == 0:
                model_classes[model_class] += 1
            data_per_model[model_class][k]["size strs"].append(size_str)
            data_per_model[model_class][k]["sizes"].append(model_size)
            data_per_model[model_class][k]["accuracies"].append(data[model_id]["mean_accuracy"])
            data_per_model[model_class][k]["stds"].append(data[model_id]["std"])
            for template_str, template_acc in data[model_id]["template_results"].items():
                sized_model = f"{model_class}-{size_str}"
                data_per_model_per_template[sized_model][k]["templates"].append(template_str.split("_")[-1])
                data_per_model_per_template[sized_model][k]["sizes"].append(model_size)
                data_per_model_per_template[sized_model][k]["accuracies"].append(template_acc)

    # Make table per model
    for model_class, num_sizes in model_classes.items():
        # model_table = np.zeros((num_sizes, len(k_shot)))
        model_table = {}
        model_tables_per_size = [{"Template": []} for _ in range(num_sizes)]
        model_sorted_size_strs = []
        for i, k in enumerate(k_shot):
            accuracies = data_per_model[model_class][k]["accuracies"]
            stds = data_per_model[model_class][k]["stds"]
            sizes = data_per_model[model_class][k]["sizes"]
            size_strs = data_per_model[model_class][k]["size strs"]
            sorted_accuracies = [x for _, x in sorted(zip(sizes, accuracies), reverse=False)]
            sorted_stds = [x for _, x in sorted(zip(sizes, stds), reverse=False)]
            sorted_sizes = sorted(sizes, reverse=False)
            sorted_size_strs = [x for _, x in sorted(zip(sizes, size_strs), reverse=False)]
            if i == 0:
                model_table["Model size"] = sorted_size_strs
            model_table[f"k = {k}"] = [f"{acc:.1f} +/- {std:.1f}" for acc, std in zip(sorted_accuracies, sorted_stds)]
            for size_i in range(len(sorted_size_strs)):
                size_str = sorted_size_strs[size_i]
                sized_model = f"{model_class}-{size_str}"
                if len(model_tables_per_size[size_i]["Template"]) == 0:
                    model_tables_per_size[size_i]["Template"] = data_per_model_per_template[sized_model][k]["templates"]
                model_tables_per_size[size_i][f"k = {k}"] = [float(f"{acc:.3}") for acc in data_per_model_per_template[sized_model][k]["accuracies"]]
                model_sorted_size_strs.append(size_str)
        for i, (sized_table, size_str) in enumerate(zip(model_tables_per_size, model_sorted_size_strs)):
            for j, col in enumerate(sized_table):
                if j == 0:
                    sized_table[col].extend(["Mean", " -- std", "Structured", " -- std", "Natural", " -- std"])
                else:
                    total_mean = np.mean(sized_table[col])
                    natural_accs = [sized_table[col][1], sized_table[col][4], sized_table[col][5]]
                    structured_accs = [sized_table[col][0], sized_table[col][2], sized_table[col][3]]
                    natural_mean = np.mean(natural_accs)
                    structured_mean = np.mean(structured_accs)
                    sized_table[col].extend([f"{total_mean:.3}",
                                             f"{np.std(sized_table[col]):.3}",
                                             f"{structured_mean:.3}",
                                             f"{np.std(structured_accs):.3}",
                                             f"{natural_mean:.3}",
                                             f"{np.std(natural_accs):.3}",])
            print(f"{model_class}-{size_str}")

            df = pd.DataFrame(sized_table)
            sized_model = f"{model_class}-{size_str}"
            # df = df.assign(mean=df.iloc[seeds, 1:].mean(axis=1))
            # df = df.assign(mean=df.mean(axis=0))
            print(df.to_latex(index=False,
                              position="ht",
                              label=f"appendix:tab:{sized_model}",
                              caption=f"Accuracy per prompt template for {sized_model}.",
                              bold_rows=True,
                              column_format="c" * (len(k_shot) + 1)))
        # df = pd.DataFrame(model_table)
        # # df = df.set_index("Model size")
        # # print(f"-------------------------- {model_class} --------------------------")
        # print(df.to_latex(index=False,
        #                   position="ht",
        #                   label=f"appendix:tab:{model_class}",
        #                   caption=f"Average accuracy for {model_class} model.",
        #                   bold_rows=True,
        #                   column_format="c" * (len(k_shot) + 1)))


def save_compute_emission_per_experiment():
    folder = "Results per model and compute"
    file_prefix = "emissions-"
    table = {"model": []}
    model_order = ["EleutherAI", "BLOOM", "OPT", "BlenderBot", "T0", "Flan-T5"]
    total_duration_per_gpu_type = defaultdict(int)
    total_duration_per_cpu_type = defaultdict(int)
    for model_folder in model_order:
        cur_path = os.path.join(*[folder, model_folder])
        if not os.path.isdir(cur_path):
            continue
        for size_folder in os.listdir(cur_path):
            cur_path = os.path.join(*[folder, model_folder, size_folder])
            if not os.path.isdir(cur_path):
                continue
            ks = []
            infos_per_k = []
            for file in os.listdir(cur_path):
                if file.startswith(file_prefix):
                    with open(os.path.join(*[folder, model_folder, size_folder, file]), "r") as infile:
                        model = model_folder
                        size = size_folder
                        k = file.split("-shot")[-2].split("-")[-1]
                        ks.append(int(k))
                        contents = infile.readlines()
                        labels = contents[0].split(",")
                        values = contents[1].split(",")
                        total_duration = 0
                        for all_values in contents[1:]:
                            these_values = all_values.split(",")
                            gpu_model = these_values[23].split("x")[-1]
                            if not gpu_model:
                                continue
                            count = int(these_values[22])
                            total_duration += float(these_values[3])
                            total_duration_per_gpu_type[gpu_model] += count * float(these_values[3])
                        keep_idxs = [0, 3, 4, 6, 7, 20, 21, 22, 23]
                        infos = {labels[i]: values[i] for i in keep_idxs}
                        infos["duration"] = total_duration
                        infos["emissions"] = f"{float(infos['emissions']):.2e}"
                        infos["k"] = int(k)
                        infos_per_k.append(infos)
            sorted_infos = [x for _, x in sorted(zip(ks, infos_per_k))]
            sorted_ks = sorted(ks)
            for i, info in enumerate(sorted_infos):
                k = sorted_ks[i]
                assert info["k"] == k
                model_name = f"{model}-{size}-{k}-shot"
                table["model"].append(model_name)
                for key, value in info.items():
                    if key not in table:
                        table[key] = []
                    table[key].append(value)

    df_1 = pd.DataFrame({"model": table["model"], "timestamp": table["timestamp"],
                         "duration": table["duration"]})
    df_2 = pd.DataFrame({"model": table["model"], "cpu_count": table["cpu_count"],
                         "cpu_model": table["cpu_model"], "gpu_model": table["gpu_model"]})
    # df = df.assign(mean=df.iloc[seeds, 1:].mean(axis=1))
    # df = df.assign(mean=df.mean(axis=0))
    print(df_1.to_latex(index=False,
                      position="ht",
                      label=f"appendix:tab:emissions",
                      caption=f"Timestamp, duration, and emissions per experiment with non-API models.",
                      bold_rows=True,
                      column_format="lll"))
    print(df_2.to_latex(index=False,
                        position="ht",
                        label=f"appendix:tab:compute",
                        caption=f"Compute used per experiment with non-API models.",
                        bold_rows=True,
                        column_format="llll"))
    for gpu_type, total_duration in total_duration_per_gpu_type.items():
        print(f"Total duration for gpu {gpu_type} is {total_duration/60/60} hours.")


def save_timestamp_per_api_call():
    # models = ["ada", "babbage", "curie", "davinci",
    #           "gpt-ada", "gpt-babbage", "gpt-curie", "gpt-3",
    #           "instructgpt",
    #           "cohere-small", "cohere-medium", "cohere-large", "cohere-xl",
    #           ]
    # names = ["GPT-3-ada", "GPT-3-babbage", "GPT-3-curie", "GPT-3-davinci",
    #          "OpenAI-ada", "OpenAI-babbage", "OpenAI-curie", "OpenAI-davinci-001",
    #          "OpenAI-Davinci-002",
    #          "Cohere-small", "Cohere-medium", "Cohere-large", "Cohere-xl",
    #          ]
    models = ["openai-text-davinci-003", "openai-chatgpt", "openai-gpt4", "cohere-commandmedium", "cohere-commandxl"]
    names = ["OpenAI-text-davinci-003", "OpenAI-gpt-3.5.turbo", "OpenAI-gpt-4", "Cohere-command-medium", "Cohere-command-xl"]
    k = [0, 1, 5, 10, 15, 30]
    timestamp_table = {"model": [], "timestamp": []}
    year = "2023"
    for i, model in enumerate(models):
        for k_shot in k:
            results_folder = f"results/{model}/{k_shot}-shot"
            files_in_dir = [file for file in os.listdir(results_folder) if file.startswith("results")]
            assert len(files_in_dir) == 1, f"Found {len(files_in_dir)} result files in {results_folder} instead of 1."
            results_file = files_in_dir.pop()
            timestamp_table["model"].append(f"{names[i]}/{k_shot}-shot")
            split_file = results_file.split("_")
            timestamp = [stamp for stamp in split_file if year in stamp]
            if not len(timestamp):
                for file in os.listdir(results_folder):
                    if year in file and ("results" in file or "data" in file):
                        split_file = file.split("_")
                        timestamp = [stamp for stamp in split_file if year in stamp]
            timestamp_table["timestamp"].append(timestamp.pop().split(".json")[0].split(".")[0])
    df = pd.DataFrame(timestamp_table)
    # df = df.assign(mean=df.iloc[seeds, 1:].mean(axis=1))
    # df = df.assign(mean=df.mean(axis=0))
    print(df.to_latex(index=False,
                      position="ht",
                      label=f"appendix:tab:timestamps",
                      caption=f"Timestamp each model behind an API was evaluated.",
                      bold_rows=True,
                      column_format="ll"))


def make_type_label_plot(project_folder):

    with open(f"{project_folder}/all_type_label_results.json", "r") as infile:
        all_results = json.load(infile)

    with open(f"{project_folder}/all_type_label_results.json", "r") as infile:
        all_results_cot = json.load(infile)

    base_models = ["Cohere", "BLOOM", "OPT", "GPT-3", "EleutherAI"]
    hf_models = ["Cohere-command", "text-<engine>-001", "text-davinci-002", "text-davinci-003",
                 "ChatGPT", "GPT-4"]
    conv_models = ["BlenderBot"]
    mt_ft_models = ["T0", "Flan-T5"]
    humans = ["Humans"]
    model_grouping = {
        "base": base_models,
        "hf": hf_models,
        "conv": conv_models,
        "mt": mt_ft_models,
        "humans": humans
    }
    models_for_line = ["Cohere-command", "GPT-4"]
    names_for_line = ["Cohere-command-52B", "GPT-4"]
    line = {"Particularised": {"x": [], "y": [], "yerr": []},
            "Generalised": {"x": [], "y": [], "yerr": []}}
    human_particularised = all_results["Humans"]["Particularised"]
    human_generalised = all_results["Humans"]["Generalised"]
    lines_per_model = {model: copy.deepcopy(line) for model in models_for_line}

    for grouping in model_grouping:
        print("--" * 10)
        print(f"Model group {grouping}")
        print("--" * 10)
        mean_diff_per_k = {f"{k}": 0 for k in [0, 1, 5, 10, 15, 30]}
        count_per_k = {f"{k}": 0 for k in [0, 1, 5, 10, 15, 30]}
        for model, results_per_k in all_results.items():
            if model not in model_grouping[grouping] or model == "Humans":
                continue
            for k, results_per_size in results_per_k.items():
                largest_model = list(results_per_size.keys())[-1]
                for size, results in results_per_size.items():
                    if size not in largest_model:
                        continue
                    diff = all_results["Humans"]["Generalised"] - results["Generalised"]["mean"]
                    mean_diff_per_k[k] += diff
                    count_per_k[k] += 1
                    if model in models_for_line:
                        lines_per_model[model]["Generalised"]["x"].append(int(k))
                        lines_per_model[model]["Generalised"]["y"].append(results["Generalised"]["mean"])
                        lines_per_model[model]["Generalised"]["yerr"].append(results["Generalised"]["std"])
                        lines_per_model[model]["Particularised"]["x"].append(int(k))
                        lines_per_model[model]["Particularised"]["y"].append(results["Particularised"]["mean"])
                        lines_per_model[model]["Particularised"]["yerr"].append(results["Particularised"]["std"])
        for k in mean_diff_per_k:
            if count_per_k[k] > 0:
                print(f"Mean difference for {k}-shot: {mean_diff_per_k[k] / count_per_k[k]}")
        print("--" * 10)
        print("--" * 10)
    mean_diff_cot = 0
    count_cot = 0
    for model, results_per_k in all_results_cot.items():
        if model == "Humans":
            continue
        for k, results_per_size in results_per_k.items():
            for size, results in results_per_size.items():
                diff = results["Generalised"]["mean"] - results["Particularised"]["mean"]
                mean_diff_cot += diff
                count_cot += 1
    print(f"Mean difference for CoT: {mean_diff_cot / count_cot}")

    models_to_add = ["text-<engine>-001", "text-davinci-002", "text-davinci-003",
                     "ChatGPT", "GPT-4", "Cohere-command"]
    model_names = ["text-davinci-001-?", "text-davinci-002-?", "text-davinci-003-?",
                     "ChatGPT-?", "GPT-4-?", "Cohere-command-52b"]
    sizes = ["unknown", "unknown", "unknown", "unknown", "unknown", "52b"]
    models_to_add = ["GPT-4", "Cohere-command"]
    model_names = ["GPT-4-?", "Cohere-command-52b"]
    sizes = ["unknown", "52b"]
    k_to_show = ["0", "5", "0"]
    plots = ["0-shot", "5-shot", "5-shot CoT"]
    k_to_show = ["0", "5"]
    plots = ["0-shot", "5-shot"]
    bars = ["Other", "Generalised", "Particularised"]
    bar_colors = {
        "Other": MODEL_COLORS["Cohere-command"],
        "Generalised": MODEL_COLORS["BLOOM"],
        "Particularised": MODEL_COLORS["Cohere"],
    }

    means_per_bar_per_plot = {plot: {bar: [] for bar in bars} for plot in plots}
    stds_per_bar_per_plot = {plot: {bar: [] for bar in bars} for plot in plots}
    for i, model in enumerate(models_to_add):
        size_of_model = sizes[i]
        plot_results = {}
        for j, k in enumerate(k_to_show):
            results = all_results[model][k][size_of_model]
            plot_results[plots[j]] = results
        for plot in plots:
            for bar in bars:
                means_per_bar_per_plot[plot][bar].append(plot_results[plot][bar]["mean"])
                stds_per_bar_per_plot[plot][bar].append(plot_results[plot][bar]["std"])

    for plot in plots:
        groups = models_to_add
        means_per_bar = means_per_bar_per_plot[plot]
        stds_per_bar = stds_per_bar_per_plot[plot]
        x = np.arange(len(groups))
        width = 0.25
        multiplier = 0

        fig, ax = plt.subplots(layout='constrained')

        for attribute, measurement in means_per_bar.items():
            std = stds_per_bar[attribute]
            offset = width * multiplier
            color = bar_colors[attribute]
            rects = ax.bar(x + offset, measurement, width,
                           yerr=std,label=attribute, color=color)
            # ax.bar_label(rects, padding=3)
            multiplier += 1

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Accuracies for "{plot}" evaluation of the group HF FT.')
        ax.set_xticks(x + width, model_names)
        plt.xticks(rotation=25)
        ax.legend(loc='upper center', ncols=3)
        ax.set_ylim(0, 100)

        plt.tight_layout()
        plt.savefig(f"{project_folder}/type_labels_barchart_{plot}.jpg")

    # Absolute plot
    linewidth = 3
    markersize = 10
    fig, axs = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(28, 14), dpi=600)
    legend_lines, legend_labels = [], []
    x_min = float("inf")
    x_max = 0
    saved_colors = {}
    colors = plt.cm.Dark2(np.linspace(0, 1, 2))
    for i, model in enumerate(models_for_line):
        # plt.figure(figsize=(16, 14), dpi=200)
        # ax = axs[int(i // 2), i % 2]
        ax = axs[i % 2]
        for j, (line_key, line_data) in enumerate(lines_per_model[model].items()):
            if line_key not in ["Particularised", "Generalised", "Other", "Mean"]:
                continue
            color = colors[j]
            saved_colors[line_key] = color
            line = ax.errorbar(
                line_data["x"],
                line_data["y"],
                yerr=line_data["yerr"],
                label=f"Prompt template {line_key}",
                marker="o",
                color=colors[j],
                linestyle="solid" if line_key != "Mean" else "dotted",
                markersize=markersize,
                linewidth=linewidth,
            )
            if min(line_data["x"]) < x_min:
                x_min = min(line_data["x"])
            if max(line_data["x"]) > x_max:
                x_max = max(line_data["x"])
            if i == 0:
                legend_lines.append(line)
                if line_key == "Mean":
                    line_key = "All types"
                legend_labels.append(f"{line_key}")
        ax.xaxis.set_tick_params(labelsize=30)
        ax.yaxis.set_tick_params(labelsize=30)
        ax.title.set_text(f"{names_for_line[i]}")
        ax.title.set_size(40)
        ax.set_ylim(45., 100.)
        if i == 0:
            ax.set_ylabel("Accuracy (%)", fontsize=45)
        ax.set_xlabel("k", fontsize=45)
    # for i in range(len(axs)):
    #     ax = axs[i, 0]
        randomline = ax.hlines(
            y=50.0,
            xmin=x_min,
            xmax=x_max,
            label="Random chance",
            linestyles="dotted",
            color="red",
            linewidth=linewidth,
        )
        part_human = ax.hlines(y=human_particularised, xmin=x_min, xmax=x_max, label="Human particularised", linestyles="dotted",
                   color=saved_colors["Particularised"], linewidth=linewidth)

        gen_human = ax.hlines(y=human_generalised, xmin=x_min, xmax=x_max, label="Human generalised", linestyles="dotted",
                   color=saved_colors["Generalised"], linewidth=linewidth)
    legend_lines.append(part_human)
    legend_labels.append("Human particularised")
    legend_lines.append(gen_human)
    legend_labels.append("Human generalised")
    legend_lines.append(randomline)
    legend_labels.append("Random chance")
    axs[1].legend(legend_lines, legend_labels, fontsize=40, loc="lower right")
    plt.suptitle(
        f"Accuracy for two different types of implicature.\nNB: Y-axis starts at 45% (just below random performance).", fontsize=50
    )
    plt.subplots_adjust(top=0.75)
    plt.tight_layout()
    plt.savefig(f"{project_folder}/type_labels_plot_absolute_new.jpg")


def api_variance(folder: str):

    likelihood_per_coherent_example = defaultdict(lambda: defaultdict(list))
    likelihood_per_incoherent_example = defaultdict(lambda: defaultdict(list))
    files_read = []
    coherent_examples = defaultdict(lambda: defaultdict(str))
    incoherent_examples = defaultdict(lambda: defaultdict(str))
    for file in os.listdir(folder):
        if "results" in file and ".json" in file:
            files_read.append(f"{folder}/{file}")
            with open(f"{folder}/{file}", "r") as infile:
                results = json.load(infile)
                for i, result in enumerate(results["predictions"]):
                    model_results = result["openai-davinci"]
                    for prompt_template, template_result in model_results.items():
                        if "prompt_template" in prompt_template:
                            correct_score = template_result["implicature_result"]["correct_score"]
                            false_score = template_result["implicature_result"]["false_score"]
                            correct_text = template_result["implicature_result"]["scored_texts"]["correct"]
                            false_text = template_result["implicature_result"]["scored_texts"]["false"]
                            if coherent_examples[i][prompt_template]:
                                assert coherent_examples[i][prompt_template] == correct_text
                                assert incoherent_examples[i][prompt_template] == false_text
                            else:
                                coherent_examples[i][prompt_template] = correct_text
                                incoherent_examples[i][prompt_template] = false_text
                            likelihood_per_coherent_example[i][prompt_template].append(-1*np.log(correct_score))
                            likelihood_per_incoherent_example[i][prompt_template].append(-1*np.log(false_score))

    print(f"Read results from following files: {files_read}")
    variance_per_example_per_template_coherent = defaultdict(lambda: defaultdict(float))
    variance_per_example_per_template_incoherent = defaultdict(lambda: defaultdict(float))

    variance_per_template_coherent = defaultdict(list)
    variance_per_template_incoherent = defaultdict(list)

    for example_idx, likelihoods_per_template in likelihood_per_coherent_example.items():
        for template, likelihoods in likelihoods_per_template.items():
            variance_per_example_per_template_coherent[example_idx][template] = np.std(likelihoods)
        for template, example_variance in variance_per_example_per_template_coherent[example_idx].items():
            variance_per_template_coherent[template].append(example_variance)

    for example_idx, likelihoods_per_template in likelihood_per_incoherent_example.items():
        for template, likelihoods in likelihoods_per_template.items():
            variance_per_example_per_template_incoherent[example_idx][template] = np.std(likelihoods)
        for template, example_variance in variance_per_example_per_template_incoherent[example_idx].items():
            variance_per_template_incoherent[template].append(example_variance)

    average_variance_per_template_coherent = defaultdict(float)
    average_variance_per_template_incoherent = defaultdict(float)
    variance_coherent_overall = 0
    variance_incoherent_overall = 0
    num_templates = 0
    for template, variance in variance_per_template_coherent.items():
        average_variance_per_template_coherent[template] = np.mean(variance)
        print(f"Coherent variance for template {template} is {average_variance_per_template_coherent[template]}")
        num_templates += 1
        variance_coherent_overall += average_variance_per_template_coherent[template]
    for template, variance in variance_per_template_incoherent.items():
        average_variance_per_template_incoherent[template] = np.mean(variance)
        print(f"Incoherent variance for template {template} is {average_variance_per_template_incoherent[template]}")
        variance_incoherent_overall += average_variance_per_template_incoherent[template]

    print(f"Variance coherent overall is {variance_coherent_overall/num_templates}")
    print(f"Variance incoherent overall is {variance_incoherent_overall / num_templates}")
    print(f"Variance overall is {(variance_coherent_overall/num_templates) + (variance_incoherent_overall / num_templates) / 2}")


def human_error_analysis():
    file = "data/type_labels.csv"
    with open(file, "r") as infile:
        file_reader = csv.reader(infile)
        keys = []
        examples_with_type_labels = {}
        for i, row in enumerate(file_reader):
            if i == 0:
                keys.extend(row)
            else:
                example = {
                    "utterance": row[0],
                    "response": row[1],
                    "implicature": row[2].strip(".").lower(),
                    "label": row[3],
                    "factual": row[4],
                }
                examples_with_type_labels[row[0].replace("\r", "")] = example
    human_eval_files = [
        "data/human_eval/human_eval - 1-150.csv",
        "data/human_eval/human_eval - 151-300.csv",
        "data/human_eval/human_eval - 301-450.csv",
        "data/human_eval/human_eval - 451-600.csv",
    ]
    human_examples, human_counts = get_all_human_performance(human_eval_files)

    examples_with_type_labels_id = {}
    found_labels = 0
    label_dist = Counter()

    for ex_id, example in enumerate(human_examples):
        key = example["utterance"].replace("\r", "")
        if key in examples_with_type_labels:
            example_labeled = examples_with_type_labels[
                example["utterance"].replace("\r", "")
            ]
            found_labels += 1
        else:
            example_labeled = {"label": "Other"}
            example_labeled.update(example)
        label_dist[example_labeled["label"]] += 1
        label_dist["Mean"] += 1
        examples_with_type_labels_id[ex_id] = example_labeled

    all_correct = 0
    all_correct_labels = Counter()
    for i, (count, ex) in enumerate(zip(human_counts, human_examples)):
        type_label = examples_with_type_labels_id[i]["label"]
        if count == 4 or count == 5 or count == 3:
            print(ex)
            all_correct += 1
            all_correct_labels[type_label] += 1
    print(all_correct)
    print(all_correct_labels)

    return


def analysis_of_cot(project_folder, k_shot=[0]):
    file = "data/type_labels.csv"
    with open(file, "r") as infile:
        file_reader = csv.reader(infile)
        keys = []
        examples_with_type_labels = {}
        for i, row in enumerate(file_reader):
            if i == 0:
                keys.extend(row)
            else:
                example = {
                    "utterance": row[0],
                    "response": row[1],
                    "implicature": row[2].strip(".").lower(),
                    "label": row[3],
                    "factual": row[4],
                }
                examples_with_type_labels[row[0].replace("\r", "")] = example
    cot_results_file = "Results per model and compute/CoT/CoT-openai-gpt-4/5-shot/results.json"
    with open(cot_results_file, "r") as infile:
        completion_cot_results = json.load(infile)
    results_file = "Results per model and compute/gpt-4/5-shot/results.json"
    with open(results_file, "r") as infile:
        completion_results = json.load(infile)

    human_eval_files = [
        "data/human_eval/human_eval - 1-150.csv",
        "data/human_eval/human_eval - 151-300.csv",
        "data/human_eval/human_eval - 301-450.csv",
        "data/human_eval/human_eval - 451-600.csv",
    ]
    human_examples, human_counts = get_all_human_performance(human_eval_files)

    # Get the example IDs for the labeled examples
    examples_with_type_labels_id = {}
    found_labels = 0
    label_dist = Counter()

    for ex_id, example in enumerate(human_examples):
        key = example["utterance"].replace("\r", "")
        if key in examples_with_type_labels:
            example_labeled = examples_with_type_labels[
                example["utterance"].replace("\r", "")
            ]
            found_labels += 1
        else:
            example_labeled = {"label": "Other"}
            example_labeled.update(example)
        label_dist[example_labeled["label"]] += 1
        label_dist["Mean"] += 1
        examples_with_type_labels_id[ex_id] = example_labeled

    count_better = 0
    for id, example_with_label in examples_with_type_labels_id.items():
        correct = completion_results["predictions"][int(id)]["openai-gpt4"]["average_implicature_accuracy"]
        correct_cot = completion_cot_results["predictions"][int(id)]["openai-gpt4"]["average_implicature_accuracy"]
        if correct <= 0 and correct_cot >= 100:
            print(example_with_label)
            count_better += 1
            print(f"Normal correct: {correct}")
            print(f"CoT correct: {correct_cot}")
            completion = completion_results["predictions"][int(id)]["openai-gpt4"]["prompt_template_0"]["implicature_result"]["correct_score"]
            cot_completion = completion_cot_results["predictions"][int(id)]["openai-gpt4"]["prompt_template_0"]["implicature_result"]["correct_score"]
            print(f"Normal completion: {completion}\n\n")
            print(f"CoT completion: {cot_completion}\n\n")
            print(f"Human count: {human_counts[int(id)]}")
    print(f"Total better: {count_better}")


if __name__ == "__main__":
    project_folder = "error_analysis_preview"
    results_folder = "results_preview"
    print(f"Running error analysis on {project_folder}, "
          "change to another folder if all_results.json is located elsewhere.")
    file = f"{project_folder}/all_results.json"

    if not os.path.exists("results_preview"):
        raise ValueError("Unzip results_preview.zip and re-run this code.")

    if not os.path.exists(project_folder):
        os.mkdir(project_folder)
    # NB: before this code will run, unzip results_preview.zip

    # The below code block can be uncommented to add results from single model files to the large file with all results
    # Gather all results per models and k and add it to one big file.
    models = [
        "cohere-commandmedium",
        "cohere-commandxl",
        "openai-gpt4",
    ]
    # Add the results obtained with this repo (from OpenAI and Cohere) to all_results.json
    k = [0, 1, 5, 10, 15, 30]
    for model in models:
        for k_shot in k:
            folder = f"{results_folder}/{model}/{k_shot}-shot"
            error_analysis_per_k(folder, project_folder)

    # Add the results from the LM-eval-harness to all_results.json
    lm_eval_folders = [f"{results_folder}/opt evals"]
    for lm_eval_folder in lm_eval_folders:
        convert_lm_eval_results(lm_eval_folder, project_folder)

    label_order = ["Best human", "Avg. human", "Cohere-command", "OPT", "Random chance"]
    models_to_show = ["Cohere-command", "OPT"]

    # Uncomment to show all models in paper if all_results.json downloaded from ...
    # label_order = ["Best human", "Avg. human", "Cohere-command", "Flan-T5", "OPT", "EleutherAI", "BLOOM", "Cohere",
    #                "GPT-3", "T0", "BlenderBot", "Random chance"]
    # models_to_show = ["OPT", "BLOOM", "EleutherAI", "Cohere", "GPT-3", "T0", "BlenderBot", "Flan-T5",
    #                   "Cohere-command"]

    plot_scale_graph(file, models_to_show=models_to_show, label_order=label_order)

    # Uncomment to show all models in an accuracy v. k graph (if all_results.json downloaded from ...)
    # groups_per_model = ["Example IT", "Example IT", "Example IT", "Example IT", "Example IT",
    #                     "Base", "Base", "Base", "Base", "Base", "Benchmark IT", "Dialogue FT", "Benchmark IT",
    #                     "Example IT"]
    # models_to_show = ["GPT-4", "ChatGPT", "text-davinci-003", "text-davinci-002", "text-<engine>-001", "OPT", "BLOOM",
    #                   "EleutherAI", "Cohere", "GPT-3", "T0", "BlenderBot", "Flan-T5",
    #                   "Cohere-command"]
    # label_order = ["GPT-4", "text-davinci-002", "text-davinci-003", "text-<engine>-001", "ChatGPT", "Cohere-command",
    #                "GPT-3", "Cohere", "OPT", "Flan-T5",
    #                "BLOOM", "EleutherAI", "BlenderBot", "T0", "Random chance"]

    groups_per_model = ["Example IT", "Base"]
    label_order_fewshot = ["Cohere-command", "OPT", "Random chance"]
    models_to_show_fewshot = ["Cohere-command", "OPT"]
    plot_few_shot(file, models_to_show=models_to_show_fewshot, label_order=label_order_fewshot,
                  groups_per_model=groups_per_model)

    type_label_analysis(project_folder,
                        models_to_show=["Cohere-command",
                                        "GPT-4"], k_shot=[0, 5])
    make_type_label_plot(project_folder)

    # Some extra analyses done based on NeurIPS 2023 reviews
    human_error_analysis()