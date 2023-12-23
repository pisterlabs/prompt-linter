import json
import os
import time 
from tqdm import tqdm
import argparse
from pathlib import Path
from typing import Tuple
import pandas as pd
import torch
import transformers
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModel, AutoModelForCausalLM
import tensor_parallel as tp
import accelerate

import templates
import ContextGenerator as ContextGenerator

def get_context(benchmark: dict, template: str, data_dir: str, ntrain: int):
    question = benchmark['question']
    cg = ContextGenerator.ContextGenerator(template=template, knowledge_dir=data_dir, k=ntrain)
    context = cg.get_top_k(question)
    return context

def get_prompt(benchmark: dict, context: str = None):
    question = benchmark['question']
    category = benchmark['category']
    choices = benchmark['choices']
    ans_type = benchmark['answer_type']
    if ans_type == 'YesOrNo':
        s = templates.YESORNO.format(question=question, info=context)
    elif category in ['A2', 'A3']:
        s = templates.PHRASE.format(question=question, info=context)
    elif ans_type == 'Sentence':
        s = templates.SENTENCE.format(question=question, info=context)
    elif ans_type == 'List':
        s = templates.LIST.format(question=question, info=context)
    elif ans_type == 'MC':
        temp = ''
        temp += f"{question}\n"
        for i, choice in enumerate(choices):
            temp += f"{chr(ord('A') + i)}. {choice}\n"
        s = templates.MC.format(question=temp, info=context)
    else:
        raise NotImplementedError
    return s 

def prepare_input(tokenizer, prompts):
    input_tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True)
    for t in input_tokens:
        if torch.is_tensor(input_tokens[t]):
            input_tokens[t] = input_tokens[t].to('cuda')

    return input_tokens

def load(ckpt_dir):
    n_gpus = torch.cuda.device_count()
    tokenizer = LlamaTokenizer.from_pretrained(
        ckpt_dir,
        use_fast=False,
        padding_side="left",
    )
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1

    # we use tensor parallel for loading llama
    model = LlamaForCausalLM.from_pretrained(ckpt_dir, low_cpu_mem_usage = True, torch_dtype=torch.float16)
    model = tp.tensor_parallel(model, [i for i in range(n_gpus)]) 
    model.eval() 

    return model, tokenizer

def batch_split(prompts, batch_num):
    batch_prompts = []
    mini_batch = []
    for prompt in prompts:
        mini_batch.append(prompt)
        if len(mini_batch) == batch_num:
            batch_prompts.append(mini_batch)
            mini_batch = []
    if len(mini_batch) != 0:
        batch_prompts.append(mini_batch)
    return batch_prompts

def batch_infer(model, tokenizer, prompts):
    batch_size = 8
    answers = []
    for batch_input in tqdm(batch_split(prompts, batch_size)):
        encode_inputs = prepare_input(tokenizer, batch_input)
        outputs = model.generate(**encode_inputs, max_new_tokens=1)
        answers.extend(tokenizer.batch_decode(outputs, skip_special_tokens=True))
    answers = [answer[-1] for answer in answers]
    return answers

def main(ckpt_dir: str, param_size: str, model_type: str, template: str, 
         data_dir: str, benchmark_file: str, output_dir: str, ans_dir: str, eval_dir: str, ntrain: int,
         note: str):
    run_results = {}
    ans_filename = f"{time.time()}_{template}_{note}.json" if note else f"{time.time()}_{template}.json"
    model, tokenizer = load(ckpt_dir)

    start_time = time.time()
    with open(benchmark_file, 'r') as f:
        benchmarks = json.load(f)

    records = []
    for benchmark in benchmarks:
        context = get_context(benchmark, template, data_dir, ntrain)
        prompt = get_prompt(benchmark, context)
        print("===========================================")
        print("[QUESTION]: ", benchmark['question'])
        print("[PROMPT]: ", prompt)
        while len(tokenizer.tokenize(prompt)) + 1> 2048: # bos token
                prompt_split = prompt.split("\n\n")
                prompt_split.pop(1)
                prompt = '\n\n'.join(prompt_split)
        label = benchmark['ground_truth']
        temp = {'id': benchmark['id'], 'answer_type': benchmark['answer_type'], 'question': benchmark['question'], 'prompt': prompt, 'label': label}
        records.append(temp)
    print("===========================================")

    prompts = [record['prompt'] for record in records]
    answers = batch_infer(model, tokenizer, prompts)
    for i, record in enumerate(records):
        record['answer'] = answers[i]
        run_results[record['id']] = {'answer': record['answer'], 'label': record['label']}

    with open(os.path.join(output_dir, ans_filename), 'w') as f:
        json.dump(records, f, indent=4)
    print("===========================================")

    with open(os.path.join(ans_dir, ans_filename), 'w') as f:
        json.dump(run_results, f, indent=4)

    # eval_output = compute_metric(os.path.join(ans_dir, ans_filename))
    # for id in eval_output:
    #     print(f"{id}: {eval_output[id]}")
    # with open(os.path.join(eval_dir, ans_filename), 'w') as f:
    #     json.dump(eval_output, f, indent=4)
    # print("===========================================")

    end_time = time.time()
    print("Total run  time: ", end_time - start_time)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', type=str, required=False, help='path to the checkpoint directory', 
                        default='')
    parser.add_argument('--param_size', type=str, required=False, help='parameter size', 
                        default='7')
    parser.add_argument('--model_type', type=str, required=False, help='model type', 
                        default='llama')
    parser.add_argument('--template', type=str, required=False, help='template', 
                        default='A')
    parser.add_argument('data_dir', type=str, required=False, help='path to the data directory',
                        default='data')
    parser.add_argument('benchmark_file', type=str, required=False, help='path to the benchmark file',
                        default='benchmark/benchmark_demo.json')
    parser.add_argument('output_dir', type=str, required=False, help='path to the output directory',
                        default='llm_output')
    parser.add_argument('--ans_dir', type=str, required=False, help='path to the answer directory',
                        default='ans_label')
    parser.add_argument('--eval_dir', type=str, required=False, help='path to the eval directory',
                        default='eval')
    parser.add_argument('--ntrain', type=int, required=False, help='number of similar paragraphs to be included in the prompt',
                        default=5)
    
    # add note to the output file name
    parser.add_argument('--note', type=str, required=False, help='note', 
                        default='langchain')
    
    args = parser.parse_args()



