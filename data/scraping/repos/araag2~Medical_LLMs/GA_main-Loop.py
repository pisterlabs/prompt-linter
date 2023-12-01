import os
import argparse
import json
import torch
import typing
import random
import re
import GA_evaluation

from openai import OpenAI
from datetime import datetime
from auto_gptq import exllama_set_max_input_length
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score
from transformers import GPTQConfig, LlamaTokenizer, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM

if "SLURM_JOB_ID" not in os.environ:
    device = "CPU"

def safe_open_w(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return open(path, 'w', encoding='utf8')

    premise = prompt_dict["primary_premise"].replace("$primary_evidence", text_to_replace["primary_evidence"])
    if "secondary_premise" in text_to_replace:
        premise += prompt_dict["secondary_premise"].replace("$secondary_evidence", text_to_replace["secondary_evidence"])
    options = prompt_dict["options"]
    
    # "$premise \n Question: Does this imply that $hypothesis? $options"
    res = prompt_dict["baseline_prompt"].replace("$premise", premise).replace("$hypothesis", text_to_replace["hypothesis"]).replace("$options", options)

    return res

def ea_generate_prompts_OAI(model_name: str, ea_prompt: str, prompt_1: str, prompt_2: str) -> str:
    new_prompt = ea_prompt.replace("$prompt_1", prompt_1).replace("$prompt_2", prompt_2)
    client = OpenAI()

    response = client.chat.completions.create(
               model=model_name,
               messages=[
               {"role": "system", "content": "You are an intelligent model that serves to combine and create better prompts from existing prompt. Follow the instructions carrefully."},
               {"role": "user", "content": new_prompt}])
    print(response)
    return response.choices.message.content

def ea_generate_pairs(curr_parent_prompts: dict, base_prompt: dict) -> dict:
    pairs = {}
    for id_1 in curr_parent_prompts:
        content_1 = curr_parent_prompts[id_1]
        for id_2 in curr_parent_prompts:
            content_2 = curr_parent_prompts[id_2]
            if id_1 != id_2:
                pairs[id_1+"_"+id_2] = {"prompt_1" : content_1, "prompt_2" : content_2, "base_prompt" : base_prompt["base_prompt"], "primary_premise" : base_prompt["primary_premise"], "secondary_premise" : base_prompt["secondary_premise"], "options" : base_prompt["options"]}
    return pairs

def sort_prompts_by_score(prompt_metrics_1: dict, prompt_metrics_2: dict, min_precison : float, max_recall : float) -> dict:
    eligible_1 = prompt_metrics_1["precision"] >= min_precison and prompt_metrics_1["recall"] <= max_recall
    eligible_2 = prompt_metrics_2["precision"] >= min_precison and prompt_metrics_2["recall"] <= max_recall

    if eligible_1 and eligible_2:
        return eligible_1["f1"] - eligible_2["f1"]
    elif eligible_1:
        return 1
    elif eligible_2: 
        return -1
    return eligible_1["f1"] - eligible_2["f1"]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_optimize_name', type=str, help='name of the model used to fine-tune prompts for', default='TheBloke/qCammel-70-x-GPTQ')

    parser.add_argument('--model_gen_prompts_name', type=str, help='name of the model used to generate and combine prompts', default='gpt-3.5-turbo')

    used_set = "dev" # train | dev | test

    # Path to corpus file
    parser.add_argument('--dataset_path', type=str, help='path to corpus file', default="../../datasets/SemEval2023/CT_corpus.json")

    # Path to queries, qrels and prompt files
    parser.add_argument('--queries', type=str, help='path to queries file', default=f'queries/queries2023_{used_set}.json')
    parser.add_argument('--qrels', type=str, help='path to qrels file', default=f'qrels/qrels2023_{used_set}.json')
    # "prompts/T5prompts.json"
    parser.add_argument('--prompts', type=str, help='path to prompts file', default="prompts/GA_Prompts-qCammel-70B.json")

    # Evaluation metrics to use 
    #
    # Model parameters
    #
    # LLM generation parameters
    #

    # Output directory
    parser.add_argument('--output_dir', type=str, help='path to output_dir', default="outputs/")

    # GA parameters
    parser.add_argument('--n_iterations', type=str, help='number of iterations to run GA on', default="3")
    parser.add_argument('--n_prompts', type=str, help='number of prompts to generate per iteration', default="5")
    parser.add_argument('--min_precision', type=str, help='minimum precision for a prompt to be considered', default="0.50")
    parser.add_argument('--max_recall', type=str, help='maximum recall for a prompt to be considered', default="0.92")

    args = parser.parse_args()

    #model = LlamaForCausalLM.from_pretrained(args.model_optimize_name, device_map="auto")
    #model.quantize_config = GPTQConfig(bits=4, exllama_config={"version":2}, desc_act=True)
    #model = exllama_set_max_input_length(model, 4096)
    
    #tokenizer = AutoTokenizer.from_pretrained(args.model_optimize_name)

    # Load dataset, queries, qrels and prompts
    queries = json.load(open(args.queries))
    qrels = json.load(open(args.qrels))
    prompts = json.load(open(args.prompts))

    # TODO: get base metrics from file
    curr_parent_prompts = {num_prompt : prompts["parent_prompts"][num_prompt] for num_prompt in prompts["parent_prompts"]}
    for i in tqdm(range(1, int(args.n_iterations)+1)):
        # Generate new prompts from pairs of parent prompts
        possible_pairs = ea_generate_pairs(curr_parent_prompts, prompts)
        new_pairs = {}
        for pair_id in tqdm(possible_pairs):
            pair_content = possible_pairs[pair_id]
            print(f'{pair_content=}')
            print(f'{prompts["ea_prompt_force-no-remove"]=} {pair_content["prompt_1"]=} {pair_content["prompt_2"]=}')
            pair_content["new_prompt"] = ea_generate_prompts_OAI(args.model_gen_prompts_name, prompts["ea_prompt_force-no-remove"], pair_content["prompt_1"]["text"], pair_content["prompt_2"]["text"])
            new_pairs[pair_id] = pair_content
        print(f'{new_pairs=}')
        #    
        #    pair["metrics"] = GA_evaluation.full_evaluate_prompt(model, tokenizer, queries, qrels, pair["new_prompt"], args, used_set)
        #print(f'The metrics of {pair=} were {pair["metrics"]=}')

if __name__ == '__main__':
    main()