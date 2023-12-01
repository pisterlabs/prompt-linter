# %%
import openai
import numpy as np
import pandas as pd
from tqdm import tqdm
from common import *
from prompt import *


# %%
sampled_df = pd.read_csv("sampled_1000.csv")
samples = sampled_df.sample(1000)
data_cols = ["impression_id", "history", "candidate", "label"]


result_path = f"result/sampled_1000_result.csv"
score_path = f"result/metrics.csv"
metric_list = ["nDCG@5", "nDCG@10", "MRR"]
results = []

# %%
for index in tqdm(samples.index, total=len(samples)):
    line = {col: samples.loc[index, col] for col in data_cols}
    user_content = build_prompt(history=line["history"], candidate=line["candidate"])
    chat_completion = openai.ChatCompletion.create(
    model=model_name,
    messages=[{
        "role": "system",
        "content": build_instruction(),
    }, {
        "role": "user",
        "content": user_content,
    }],
    temperature=0)
    line["output"] = chat_completion['choices'][0]['message']['content']
    line.update(evaluate_output(line['output'], line["label"], metric_list))
    results.append(line)
    save2file(results, result_path)
    cal_avg_scores(results, score_path, model_name, metric_list)


# %%
