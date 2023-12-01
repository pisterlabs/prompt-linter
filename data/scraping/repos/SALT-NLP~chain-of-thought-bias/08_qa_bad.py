import openai
import json
from tqdm import tqdm
import time

with open("./data/dangerous-q/toxic_outs.json") as f:
    corpus = json.load(f)

openai.api_key = "API_KEY_HERE"

def get_completion(
    templated_prompt, 
    temp=0.7,
    max_tokens=256,
    n=5,
    model = "text-davinci-001"
):
    # while True:
    #     try:
    response = openai.Completion.create(
        model=model,
        prompt=templated_prompt,
        temperature=temp,
        max_tokens=max_tokens,
        n=n,
    )
    return [choice["text"] for choice in response["choices"]]
        # except:
        #     print("sad")
        #     time.sleep(15)
        #     continue

outs = {}
try:
    with open("./output/qa/davinci-001.json") as f:
        outs = json.load(f)
except:
    outs = {}

for k in tqdm(range(len(corpus))):
    if k in outs: continue
    norm_out = get_completion(corpus[k])
    cot_prompt = corpus[k] + " Let's think step by step."
    cot_out = get_completion(cot_prompt)
    outs[k] = {
        "norm_out": norm_out,
        "cot_out": cot_out,
        "cot_prompt": cot_prompt,
        "prompt": corpus[k]
    }


    with open("./output/qa/davinci-001.json", 'w', encoding='utf-8') as f:
        json.dump(outs, f, ensure_ascii=False, indent=4)