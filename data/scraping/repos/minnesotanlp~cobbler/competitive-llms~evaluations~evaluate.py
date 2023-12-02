from datasets import load_dataset
import numpy as np
import sys
import torch

from tqdm import tqdm
import re
import json

import random
import itertools

from model_configs import model_configs
from utils import guidance_uniform_chat, uniform_prompt_func, guidance_uniform_completion, guidance_models, get_guidance_model

from order import evaluate_order
from bandwagon import evaluate_bandwagon
from compassion import evaluate_compassion
from selective import evaluate_selective
from salience import evaluate_salience
from distraction import evaluate_distraction
from frequency import evaluate_frequency

sys.path.append('../talkative-llm')
from talkative_llm.llm import get_supported_llm

import yaml
import random
from math import comb
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
x=5

N=50

random.seed(939)

models = ["chatgpt", "instructgpt", "gpt4", 
    "cohere", "llama", "falcon",
    "openassist", "dolly", "alpaca", "baize",
    "redpajama", "koala", "vicuna", "wizardlm", "mpt",
    "random"
    ]

def evaluate_nC2(evaluator, instructions, reference, responses, human, bias_mode="order"):
    print(f"evaluating with {evaluator}")
    if evaluator != "all" and evaluator != "random":
        if evaluator not in guidance_models:
            config_path = model_configs[evaluator]
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            caller = get_supported_llm(config)
            eval_gen = caller
        else:
            prompter = get_guidance_model(evaluator, bias_mode)
            eval_gen = prompter
    elif evaluator == "random":
        eval_gen = "random"
    
    if bias_mode == "order":
        evaluate_order(N, evaluator, instructions, reference, responses, eval_gen)
    elif bias_mode == "bandwagon":
        evaluate_bandwagon(N, evaluator, instructions, reference, responses, eval_gen)
    elif bias_mode == "compassion":
        evaluate_compassion(N, evaluator, instructions, reference, responses, eval_gen)
    elif bias_mode == "selective":
        evaluate_selective(N, evaluator, instructions, reference, responses, eval_gen)
    elif bias_mode == "salience":
        if evaluator == "all":
            for model in models:
                evaluate_salience(model)
        else:
            evaluate_salience(evaluator)
    elif bias_mode == "distraction":
        evaluate_distraction(N, evaluator, instructions, reference, responses, eval_gen)
    elif bias_mode == "frequency":
        evaluate_frequency(N, evaluator, instructions, reference, responses, eval_gen)
    
        
def read_json_file(file):
    with open(file, "r") as r:
        response = r.read()
        response = response.replace('\n', '')
        response = response.replace('}{', '},{')
        response = "[" + response + "]"
        return json.loads(response)

def main(batch, bias):
    with open('../competitive-llms/datasets/llm_preference_evalset.json', 'r') as file:
        data = file.read()
        dataset = json.loads(data)

    instructions = [data['instruction'] for data in dataset]
    references = [data['reference'] for data in dataset]
    responses = read_json_file("../competitive-llms/n15_responses/full_n15_model_generations.json")[0]
    
    # Batch 1
    if batch == 0:
        evaluators = ["all"]
    if batch == 4:
        evaluators = ["random"]
    
    print(batch)
    if batch == 1:
        evaluators = ["chatgpt", "instructgpt", "gpt4", "cohere"]

    # Batch 2
    elif batch == 2:
        evaluators = ["baize", "koala", "alpaca"] #, "dolly"] "alpaca"
    
    elif batch == 3:
    # Batch 3
        evaluators = ["openassist", "wizardlm", "redpajama", "dolly"]
    
    for ranker in evaluators:
        if bias:
            evaluate_nC2(ranker, instructions, references, responses, human=None, bias_mode=bias)
        else:
            evaluate_nC2(ranker, instructions, references, responses, human=None)

if __name__ == "__main__":
    # Batch {1, 2, or 3}
    arg1 = sys.argv[1]
    # Include "human" response / ground truth
    if len(sys.argv) > 2:
        arg2 = sys.argv[2]
        print(f"Evaluating {arg2} bias")
    else:
        arg2 = None
    main(int(arg1), str(arg2))
