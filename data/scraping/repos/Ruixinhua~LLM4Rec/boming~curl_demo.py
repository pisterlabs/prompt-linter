import openai
import numpy as np
import pandas as pd
from tqdm import tqdm
from common import *
from prompt import *
import urllib.request
import json
import datetime
import sys
import re


# -------- Get Model Name -------------------
openai.api_key = "EMPTY"
openai.api_base = "http://bendstar.com:8000/v1"
models = openai.Model.list()
model_name = models["data"][0]["id"]


# --------- Samples Setting---------------------
sample_number = 100
sampled_df = pd.read_csv("sampled_1000.csv")
samples = sampled_df.sample(sample_number, random_state=42)
data_cols = ["impression_id", "history", "candidate", "label"]

dt_now_random = datetime.datetime.now().second
result_path = f"result/sampled_{sample_number}_result_{dt_now_random}.csv"
score_path = f"result/metrics_news.csv"
metric_list = ["nDCG@5", "nDCG@10", "MRR"]


# ---------- Template Setting-------------------
function_dispatcher = {
    '0': build_prompt_template0,
    '1': build_prompt_template1,
    '2': build_prompt_template2,
    '99': build_prompt_template99,
    '10': build_prompt_template10,
    '15': build_prompt_template15,
    '50': build_improved_prompt50,
    '4': build_prompt_template4,
}


# ---------- URL request Setting ----------------
url = "http://bendstar.com:8000/v1/chat/completions"
req_header = {
    'Content-Type': 'application/json',
}


if __name__ == '__main__':
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = "0"
    func = function_dispatcher.get(choice)
    
    results = []
    for index in tqdm(samples.index, total=len(samples)):
        line = {col: samples.loc[index, col] for col in data_cols}

        user_content = func(history=line['history'], candidate=line["candidate"])

        chat_completion = json.dumps({
            "model": model_name,
            "messages": [{"role": "system", "content": build_instruction()}, 
                        {"role": "user", "content": user_content}],
            "temperature": 0,
        })
        line["input"] = user_content
        req = urllib.request.Request(url, data=chat_completion.encode(), method='POST', headers=req_header)

        with urllib.request.urlopen(req) as response:
            body = json.loads(response.read())
            headers = response.getheaders()
            status = response.getcode()
            print(body['choices'][0]['message']['content'])

        
        line["output"] = body['choices'][0]['message']['content']
        line.update(evaluate_output(line['output'], line["label"], metric_list))
        results.append(line)
        save2file(results, result_path)
        result_df = cal_avg_scores(results, model_name, metric_list, choice, dt_now_random)
    
    if os.path.exists(score_path):
        result_df.to_csv(score_path, mode='a', index=False, header=False)
    else:
        result_df.to_csv(score_path, index=False)