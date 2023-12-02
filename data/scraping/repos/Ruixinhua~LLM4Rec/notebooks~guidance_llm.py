import os
import urllib.request
import guidance
import json
import metric_utils as module_metric
import numpy as np
import pandas as pd
from tqdm import tqdm
from string import Template
from common import load_api_key, load_cmd_line
from utils import convert2list, save2csv, cal_avg_scores, seed_everything, extract_output, evaluate_list, extract_raw_output
import templates
from templates import gpt_template, llama_template

url = "http://bendstar.com:8000/v1/chat/completions"
req_header = {
    'Content-Type': 'application/json',
}


def is_descending(lst):
    for i in range(len(lst) - 1):
        if lst[i] < lst[i + 1]:
            return False
    return True


def build_instruction():
    return "You serve as a personalized news recommendation system."


def run_gpt(data):
    system = guidance(template, llm=model, silent=True)
    return system(history=data["history"], candidate=data["candidate"])


def request_llama(data, temp):
    user_content = temp.replace("{{history}}", data["history"]).replace("{{candidate}}", data["candidate"])
    chat_completion = json.dumps({
        "model": "meta-llama/Llama-2-70b-chat-hf",
        "messages": [{"role": "system", "content": build_instruction()}, {"role": "user", "content": user_content}],
        "temperature": 0,
    })
    req = urllib.request.Request(url, data=chat_completion.encode(), method='POST', headers=req_header)

    with urllib.request.urlopen(req) as response:
        body = json.loads(response.read())
        print(body['choices'][0]['message']['content'])
        return body['choices'][0]['message']['content']


if __name__ == "__main__":
    # we use LLaMA here, but any GPT-style model will do
    args = load_cmd_line()
    seed_everything(args.get("seed", 42))
    # template_list = {1: template1, 2: template2, 3: template3, 4: template4, 5: template5}
    template_no = args.get("template_no", 4)
    num = args.get("num", 1000)
    data_used = args.get("data_used", "order")
    sampled_df = pd.read_csv(f"sampled_1000_{data_used}.csv")
    # sampled_df["candidate"] = sampled_df["candidate"].apply(shuffle_candidates)
    samples = sampled_df.sample(num)
    model_name = args.get("model_name", "gpt-3.5-turbo")
    temperature = args.get("temperature", 0)
    suffix = f"template-{template_no}_{model_name}_{data_used}_temperature-{temperature}"
    result_path = f"generated_data/{suffix}.csv"
    os.makedirs(f"result/sampled_{num}", exist_ok=True)
    score_path = f"result/sampled_{num}/{suffix}.csv"
    data_cols = ["impression_id", "history", "candidate", "label"]
    user_template = getattr(templates, f"template{template_no}")
    if "gpt" in model_name.lower():
        model = guidance.llms.OpenAI(model_name, api_key=load_api_key())
        template = Template(gpt_template).safe_substitute({"temperature": temperature, "input": user_template,
                                                           "max_tokens": args.get("max_tokens", 200)})
    elif "llama_request" in model_name.lower():
        template = Template(llama_template).safe_substitute({"input": user_template})
    else:
        model = guidance.llms.Transformers(f"meta-llama/{model_name}", device_map="auto")
        template = Template(llama_template).safe_substitute({"input": user_template})
    metric_list = ["group_auc", "mean_mrr", "ndcg_5", "ndcg_10"]
    # metric_list = ["nDCG@5", "nDCG@10", "MRR"]
    metric_funcs = [getattr(module_metric, met) for met in metric_list]
    results = []
    in_order_ratio = 0
    for index in tqdm(samples.index, total=len(samples)):
        line = {col: samples.loc[index, col] for col in data_cols}
        if "gpt" in model_name.lower():
            line["output"] = run_gpt(line)["rank"]
        else:
            line["output"] = request_llama(line, template)
        if data_used == "order":
            line["rank"] = ','.join(extract_output(line["output"], line["candidate"]))
        else:
            line["rank"] = ','.join(extract_raw_output(line["output"], line["candidate"]))
        output_list, label_list = convert2list(line["rank"], line["label"], line["candidate"])
        in_order_ratio += 1 if is_descending(output_list[np.nonzero(output_list)[0]]) else 0
        line.update(evaluate_list(output_list, label_list, metric_funcs))
        results.append(line)
        save2csv(results, result_path)
        cal_avg_scores(results, score_path, model_name, metric_list)
    in_order_ratio = in_order_ratio / num
    df = pd.read_csv(score_path)
    df["in_order_ratio"] = round(in_order_ratio, 3)
    df["max_tokens"] = args.get("max_tokens", 200)
    df["template_no"] = template_no
    df["temperature"] = temperature
    df["dataset"] = f"sampled_1000_{data_used}"
    df["seed"] = args.get("seed", 42)
    df.to_csv(score_path, index=False)
    print(f"the ratio of in-order output is {in_order_ratio}")
