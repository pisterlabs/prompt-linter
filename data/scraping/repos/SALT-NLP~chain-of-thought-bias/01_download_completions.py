
import pandas as pd
import openai
import json
from tqdm import tqdm
import argparse
from prompts import prompt_templates
import os
import time
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import transformers
from multiprocessing.managers import BaseManager


parser = argparse.ArgumentParser()
parser.add_argument('--data-path', help='input-data')
parser.add_argument('--output-path', help='input-data')
parser.add_argument('--prompt-strategy', help='input-data')
parser.add_argument('--prompt-path', help='input-data')
parser.add_argument('--word', help='input-data')
parser.add_argument('--model', default="text-davinci-002")
parser.add_argument('--cot', action='store_true')
parser.add_argument('--limit', help='input-data', type=int)
parser.set_defaults(cot=False)
args = parser.parse_args()

limit = args.limit
data = args.data_path
outputs = args.output_path
prompt_strategy = args.prompt_strategy
cot_mode = args.cot
word = args.word
prompt_path = args.prompt_path
model = args.model

prompt_template = prompt_templates[prompt_strategy]

flan_model = None
tokenizer = None

def get_completion(
    templated_prompt, 
    temp=0.7,
    max_tokens=256,
    n=1,
    model = "text-davinci-002"
):
    global flan_model
    global tokenizer
    
    if "flan" in model:
        if flan_model is None:
            print("LOADING FLAN FROM MANAGER")
            tokenizer = T5Tokenizer.from_pretrained(f"google/{model}")
            manager = BaseManager(('', 37844), b'flanserver')
            manager.register('get_connection')
            manager.register('get_name')
            manager.connect()
            flan_model = manager.get_connection()
            name = manager.get_name()
            name = str(name)[1:-1]
            print("MANAGER NAME")
            print(name)
            print(model)
            if name != model:
                raise Exception("Model mismatch")

        input_ids = tokenizer(templated_prompt, return_tensors="pt").input_ids.to("cuda")
        output_ids = flan_model.generate(input_ids.repeat(5, 1), max_new_tokens=256, do_sample=True, temperature=0.7, use_cache=True)
        return tokenizer.batch_decode(output_ids, skip_special_tokens=True)

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


open_mode = 'r' if os.path.exists(outputs) else 'w+'
with open(outputs, open_mode) as f:
    try:
        out_map = json.load(f)
    except: out_map = {}


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

    if cot_mode:
        prompt += prompt_template["cot_initial"]
    else: prompt += prompt_template["final"]

    prompt_map[str(i)] = prompt
    
    with open(prompt_path, 'w', encoding='utf-8') as f:
        json.dump(prompt_map, f, ensure_ascii=False, indent=4)

    if str(i) in out_map:
        continue

    if i not in out_map:
        out_map[str(i)] = []

    out_map[str(i)].extend(get_completion(prompt, n=5, model=model))

    with open(outputs, 'w', encoding='utf-8') as f:
        json.dump(out_map, f, ensure_ascii=False, indent=4)

with open(outputs, 'w', encoding='utf-8') as f:
    json.dump(out_map, f, ensure_ascii=False, indent=4)
    
with open(prompt_path, 'w', encoding='utf-8') as f:
    json.dump(prompt_map, f, ensure_ascii=False, indent=4)
