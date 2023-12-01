
import pandas as pd
import openai
import json
from tqdm import tqdm
import argparse
from prompts import prompt_templates
import os
import torch
import time
from transformers import T5Tokenizer, T5ForConditionalGeneration
from multiprocessing.managers import BaseManager

parser = argparse.ArgumentParser()
parser.add_argument('--data-path', help='input-data')
parser.add_argument('--output-path', help='input-data')
parser.add_argument('--cot-answer-path', help='input-data')
parser.add_argument('--prompt-path', help='input-data')
parser.add_argument('--prompt-strategy', help='input-data')
parser.add_argument('--word', help='input-data')
parser.add_argument('--model', default="text-davinci-002")
parser.add_argument('--limit', help='input-data', type=int)
args = parser.parse_args()

limit = args.limit
data = args.data_path
outputs = args.output_path
cot_answer_path = args.cot_answer_path
prompt_strategy = args.prompt_strategy
prompt_template = prompt_templates[prompt_strategy]
word = args.word
prompt_path = args.prompt_path
model = args.model

flan_model = None
tokenizer = None

def get_completion(
    templated_prompt, 
    temp=0.7,
    max_tokens=256,
    n=1,
    model = "text-davinci-002"
):
    while True:
        try:
            response = openai.Completion.create(
                model=model,
                prompt=templated_prompt,
                temperature=temp,
                max_tokens=max_tokens,
                n=n,
            )
            return [choice["text"] for choice in response["choices"]]
        except:
            print("sad")
            time.sleep(15)
            continue

with open(outputs) as f:
    out_map = json.load(f)

open_mode = 'r' if os.path.exists(cot_answer_path) else 'w+'
with open(cot_answer_path, open_mode) as f:
    try:
        cot_answer = json.load(f)
    except:
        cot_answer = {}


open_mode = 'r' if os.path.exists(prompt_path) else 'w+'
with open(prompt_path, open_mode) as f:
    try:
        prompt_map = json.load(f)
    except: prompt_map = {}

df = pd.read_csv(data)

if limit:
    df = df[:limit]

for i, row in tqdm(df.iterrows(), total=len(df)):
    all_choices = [
        row["a"], 
        row["b"], 
        row["c"]
    ]
    
    if "context" in row and (not pd.isna(row["context"])):
        for ix, choice in enumerate(all_choices):
            if row["sent_more"] in choice or row["sent_less"] in choice:
                all_choices[ix] = row["context"] + " " + all_choices[ix]

    question = None
    if "ctx" in row and (not pd.isna(row["ctx"])) and "q_text" in row and (not pd.isna(row["q_text"])):
        question = row["ctx"] + " " + row["q_text"]

    prompt = prompt_template["template"](all_choices, word, question=question)
    prompt += prompt_template["cot_initial"]

    prompt_map[str(i)] = []

    for idx, completion in enumerate(out_map[str(i)]):
        x = prompt + completion + prompt_template["cot_final"]
        prompt_map[str(i)].append(x)
        
        if i % 100 == 0:
            with open(prompt_path, 'w', encoding='utf-8') as f:
                json.dump(prompt_map, f, ensure_ascii=False, indent=4)

        if str(i) in cot_answer and len(cot_answer[str(i)]) == 5:
            continue

        if str(i) not in cot_answer:
            cot_answer[str(i)] = {}

        if str(idx) in cot_answer[str(i)]:
            continue
        
        cot_answer[str(i)][str(idx)] = get_completion(x, model=model)
        
        if i % 100 == 0:
            with open(cot_answer_path, 'w', encoding='utf-8') as f:
                json.dump(cot_answer, f, ensure_ascii=False, indent=4)

                
                
with open(cot_answer_path, 'w', encoding='utf-8') as f:
    json.dump(cot_answer, f, ensure_ascii=False, indent=4)
    
with open(prompt_path, 'w', encoding='utf-8') as f:
    json.dump(prompt_map, f, ensure_ascii=False, indent=4)
    
