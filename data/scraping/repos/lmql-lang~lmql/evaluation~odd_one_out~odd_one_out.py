import os
import torch
import pandas as pd
import asyncio
import math
import json
import sys
import random
import argparse
from tqdm import tqdm
from dataclasses import dataclass

from concurrent.futures import ProcessPoolExecutor

sys.path.append("../../")
sys.path.append("../")
import lmql

from utils.dc_baseline_prompter import HFPrompter
from utils.openai_baseline_prompter import OpenAIPrompter

def get_data():
    if not os.path.exists("task.json"):
        os.system('wget https://raw.githubusercontent.com/google/BIG-bench/main/bigbench/benchmark_tasks/odd_one_out/task.json')

    with open("task.json") as f:
        data = json.load(f)
    return data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("method", type=str, help="Method to evaluate.")
    parser.add_argument("--model", type=str, help="Model to evaluate.", default="EleutherAI/gpt-j-6B")
    parser.add_argument("--workers", type=int, help="Number of workers.", default=1)
    parser.add_argument("--num-samples", dest="num_samples", type=int, help="Limits the number of samples that are evaluated.", default=None)
    return parser.parse_args()

args = get_args()

global module
module = None
lmql.connect()

@dataclass
class SampleResult:
    correct_item: str
    result_item: str
    result_probs: list
    interaction_trace: str
    query: str = None

async def process_lmql(d):
    options = ", ".join(sorted(d["target_scores"].keys()))
    
    # print(f"""OPTIONS = "{options}\"""")
    # print(module.lmql_code)

    result = await module.query(options)

    if result is None or (type(result) is list and ((len(result) > 0 and result[0] is None) or len(result) == 0)):
        result_item = "ERR"
        result_probs = [(x, 0.0) for x in options.split(", ")]
        prompt = "<failed to decode full promtp>"
    else:
        result_item = result.variables.get("RESULT")
        result_probs = result.variables.get("P(RESULT)")
        prompt = result.prompt
    
    correct_item = sorted(d["target_scores"].items(), key=lambda x: x[1], reverse=True)[0][0]
    
    return SampleResult(correct_item, result_item, result_probs, prompt, None)

async def baseline(d):
    options = ", ".join(sorted(d["target_scores"].keys()))
    correct_item = sorted(d["target_scores"].items(), key=lambda x: x[1], reverse=True)[0][0]
    assert d["target_scores"][correct_item] == 1.0

    random_result = options.split(", ")[0]
    return [correct_item, random_result, options, d["target_scores"]]

async def hf_baseline(d):
    options = ", ".join(sorted(d["target_scores"].keys()))
    correct_item = sorted(d["target_scores"].items(), key=lambda x: x[1], reverse=True)[0][0]

    prompt = f"""
Pick the odd word out: skirt, dress, pen, jacket.
skirt is clothing, dress is clothing, pen is an object, jacket is clothing.
So the odd one is pen.

Pick the odd word out: Spain, France, German, England, Singapore.
Spain is a country, France is a country, German is a language, England is a country, Singapore is a country.
So the odd one is German.

Pick the odd word out: {options}""".strip() + "\n"

    hf = HFPrompter(args.model)
    
    prompt_with_reasoning = await hf.generate(prompt, max_new_tokens=40, stopping_phrases=["So the odd one is", "Pick the odd word out"], step_size=20)
    reasoning = prompt_with_reasoning[len(prompt):].strip()

    res = await hf.cond_logprob(prompt_with_reasoning.strip(), options.split(", "))
    result_item = max(res, key=lambda x: x[1])[0]
    probs = [(x[0], math.exp(x[1])) for x in res]
    # print(res, hf.last_num_steps)
    
    return SampleResult(correct_item, result_item, probs, prompt_with_reasoning.strip() + result_item)

async def openai_baseline(d):
    options = ", ".join(sorted(d["target_scores"].keys()))
    correct_item = sorted(d["target_scores"].items(), key=lambda x: x[1], reverse=True)[0][0]

    prompt = f"""
Pick the odd word out: skirt, dress, pen, jacket.
skirt is clothing, dress is clothing, pen is an object, jacket is clothing.
So the odd one is pen.

Pick the odd word out: Spain, France, German, England, Singapore.
Spain is a country, France is a country, German is a language, England is a country, Singapore is a country.
So the odd one is German.

Pick the odd word out: {options}.
    """.strip() + "\n"

    hf = OpenAIPrompter(args.model)
    
    prompt_with_reasoning = await hf.generate(prompt, max_new_tokens=40, stopping_phrases=["So the odd one is", "Pick the odd word out"], step_size=20)
    reasoning = prompt_with_reasoning[len(prompt):].strip()
    
    print([prompt_with_reasoning.strip()+"(" + "|".join(options.split(", ")) + ")"])

    res = await hf.cond_logprob(prompt_with_reasoning.strip(), options.split(", "))
    result_item = max(res, key=lambda x: x[1])[0]
    probs = [(x[0], math.exp(x[1])) for x in res]
    # print(res, hf.last_num_steps)

    return [correct_item, result_item, reasoning, options, probs]

async def process_sample_task(d, sem):
    async with sem:
        if d["model"] == 'baseline':
            return await baseline(d)
        elif d["model"] == "hf":
            return await hf_baseline(d)
        if d["model"] == 'oai':
            return await openai_baseline(d)
        elif d["model"].endswith(".lmql"):
            return await process_lmql(d)
        else:
            assert False, "Unknown model {}".format(d["model"])

def result_file(file):
    model_name_suffix = ""
    
    model_name_suffix = f"-{args.model}"
    model_name_suffix = model_name_suffix.replace("/", "-")
    
    if args.num_samples is not None:
        model_name_suffix += f"-n{args.num_samples}"

    return f"results/{file}-{args.method}{model_name_suffix}.txt"

class PrintingDebuggerOutputWriterWithStats:
    def __init__(self, only_description=False):
        self.stats = ""
        self.only_description = only_description

    def set_description(self, d):
        self.stats = d
        if self.stats.strip() != "":
            print(self.stats)


    def add_interpreter_head_state(self, variable, head, prompt, where, trace, is_valid, is_final, mask, num_tokens, program_variables):
        pass
        # if head == 0:
            # os.system("clear")
            # print(f"{prompt}\n\n valid={is_valid}, final={is_final}{self.stats}")
    def add_compiler_output(self, code): pass

async def run_eval():
    model = args.method
    print(f"Running with {model}")
    data = get_data()

    global module
    if module is None and model.endswith(".lmql"):
        module = lmql.load(model, autoconnect=True, force_model=args.model)
        sem = asyncio.Semaphore(args.workers)
        progress_output_writer = PrintingDebuggerOutputWriterWithStats()
        module.query.output_writer = progress_output_writer
    elif "oai" == args.method:
        lmql.autoconnect()
        sem = asyncio.Semaphore(args.workers)
        progress_output_writer = PrintingDebuggerOutputWriterWithStats(only_description=True)
    else:
        sem = asyncio.Semaphore(args.workers)
        lmql.autoconnect()
        progress_output_writer = PrintingDebuggerOutputWriterWithStats(only_description=True)
        # for other baseline do not process in parallel

    results = []

    n = 0
    n_correct = 0.0

    results = []
    data = data["examples"]
    for d in data: 
        d["model"] = model
        
    if args.num_samples is not None:
        offset = 48
        data = data[offset:offset+args.num_samples]

    served_model = lmql.model_registry.get(args.model).served_model
    if hasattr(lmql.model_registry.get(args.model), "hf_stats"):
        served_model = lmql.model_registry.get(args.model).hf_stats
    served_model.reset_stats()

    for result in asyncio.as_completed([process_sample_task(d, sem) for d in data]):
        result = await result

        n += 1
        results += [result]

        result_item = result.result_item
        correct_item = result.correct_item
        n_correct += 1 if (result_item == correct_item) else 0
        
        # print("\t".join([str(x) for x in result]))
        
        df = pd.DataFrame([r.__dict__ for r in results])
        df.to_csv(result_file("results.csv"), index=False)

        progress_output_writer.set_description("\n {}/{} Accuracy: {:.2f}, Queries: {}, Tokens/Prompt: {}".format(n, len(data), n_correct / max(1,n), served_model.num_queries, served_model.consumed_tokens / max(1, n)))

        with open(result_file("score"), "w") as f:
            f.write("Accuracy: {:.2f}\n".format(n_correct / max(1,n)))
            f.write("Queries: {}\n".format(served_model.num_queries))
            f.write("Tokens/Prompt: {}\n".format(served_model.consumed_tokens))
            f.write("generate() calls: {}\n".format(served_model.num_generate_calls))
            f.write("billable tokens: {}\n".format(served_model.billable_tokens))

if __name__ == "__main__":
    asyncio.run(run_eval())