import os
import argparse
import openai
import math
from tqdm import tqdm

from utils import *
from dataset_utils import read_synth_data, index_example, reorder_rationale
from joint import prompt_for_joint_prediction, is_factual, evaluate_joint_predictions
import numpy as np

openai.api_key = os.getenv("OPENAI_API_KEY")

def _parse_args():
    parser = argparse.ArgumentParser()
    add_engine_argumenet(parser)

    # standard, instruction, etc
    parser.add_argument('--style', type=str, default="p-e", choices=["p-e", "p-e-r"])
    parser.add_argument('--run_prediction', default=False, action='store_true')
    parser.add_argument('--num_shot', type=int, default=16)
    parser.add_argument('--train_slice', type=int, default=0)
    parser.add_argument('--num_dev', type=int, default=250)
    parser.add_argument('--strategy', type=str, default="random", choices=["random"])
    parser.add_argument('--reorder', default=False, action='store_true')
    
    args = parser.parse_args()
    specify_engine(args)
    return args

def result_cache_name(args):
    return "misc/score_joint_{}_tr{}-{}_dv{}_{}_predictions.json".format(args.engine_name,
                    args.train_slice, args.train_slice + args.num_shot, args.num_dev, args.style)

def get_candidate_answers(ex):
    context = ex["context"]    
    clues = context.strip(".").split(",")
    candidates = [x.split()[0] for x in clues[0]]    
    return candidates

def parse_answer_and_rationale(text, style):
    text = text.strip()

    # place holder
    answer = "null"
    rationale = "null"
    
    if style == "p-e-r":
        sep = ', because '
        fields = text.split(sep)
        if len(fields) != 2:
            print("outlier", fields)
        answer, rationale = fields[0], fields[1]
    elif style == "p-e":
        sep = ', because '
        fields = text.split(sep)
        if len(fields) != 2:
            print("outlier", fields)
        answer, rationale = fields[0], fields[1]
    else:
        raise RuntimeError("Unsupported decoding style")
            
    return answer, rationale

def get_candidate_answers(pred):
    completion_offset = pred["completion_offset"]
    token_offset = pred["logprobs"]["text_offset"]
    tokens = pred["logprobs"]["tokens"]
    completion_start_tok_idx = token_offset.index(completion_offset)

    branches = pred['logprobs']['top_logprobs'][completion_start_tok_idx]

    branches = sorted([x for x in branches], key=lambda x: branches[x], reverse=True)

    branches = [x for x in branches if x.strip() and x[0] == " " and x.strip()[0].isupper()]
    top_branch = pred['logprobs']['tokens'][completion_start_tok_idx]
    assert top_branch == branches[0]
    return branches, top_branch

    
def in_context_candidates_scoring(ex, shots, engine, style="standard", strategy="random", num_shot=10):
    print("---------------------")
    prompt, stop_signal = prompt_for_joint_prediction(ex, shots, style)    
    resp = openai.Completion.create(engine=engine, prompt=prompt, temperature=0.0, max_tokens=48, echo=True, logprobs=5, stop=stop_signal)
    # resp = read_json("temp.json")
    pred = resp["choices"][0]
    pred["text"] = pred["text"][len(prompt):]
    pred["completion_offset"] = len(prompt)
    
    natural_pred = pred
    natural_ans, natural_rationale = parse_answer_and_rationale(natural_pred["text"], style) 
    # get candidates
    answer_candidates, top_branch = get_candidate_answers(pred)
    print("GT", ex["answer"], ex["text_rationale"])
    for candidate in answer_candidates:
        if candidate == top_branch:
            candidate_pred = natural_pred
        else:
            can_prompt = prompt + candidate

            resp = openai.Completion.create(engine=engine, prompt=can_prompt, temperature=0.0, max_tokens=48, echo=True, logprobs=5, stop=stop_signal)
            candidate_pred = resp["choices"][0]
            candidate_pred["text"] = candidate_pred["text"][len(prompt):]
            candidate_pred["completion_offset"] = len(prompt)
        candidate_ans, candidate_rationale = parse_answer_and_rationale(candidate_pred["text"], style) 
        if is_factual(candidate_rationale, ex["context"]):
            print(candidate_ans, "|", candidate_rationale, candidate_ans==ex["answer"], "Factual")
            return candidate_pred
        else:
            print(candidate_ans, "|", candidate_rationale, candidate_ans==ex["answer"], "Nonfact")
    return natural_pred

def conditional_strip_prompt_prefix(x, p):
    if x.startswith(p):
        x = x[len(p):]
    return x.strip()

def test_few_shot_candidates_scoring(args):
    print("Running prediction")
    train_set = read_synth_data("data/100-train_synth.json")
    dev_set = read_synth_data("data/250-dev_synth.json")

    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]
    dev_set = dev_set[:args.num_dev]
    
    train_set = [index_example(x) for x in train_set]
    dev_set = [index_example(x) for x in dev_set]
    predictions = []
    for x in tqdm(dev_set, total=len(dev_set), desc="Predicting"):
        predictions.append(in_context_candidates_scoring(x, train_set, engine=args.engine, style=args.style, strategy=args.strategy, num_shot=args.num_shot))    
    # save
    # read un indexed dev
    dump_json(predictions, result_cache_name(args))        
    predictions = [process_joint_prediction(p, args.style) for p in predictions]
    # acc
    evaluate_joint_predictions(dev_set, predictions)

def process_joint_prediction(p, style):
    text = p["text"]
    text = text.strip()

    # place holder
    answer = "null"
    rationale = "null"
    
    if style == "p-e-r":
        sep = ', because '
        fields = text.split(sep)
        if len(fields) != 2:
            print("outlier", fields)
        answer, rationale = fields[0], fields[1]
    elif style == "p-e":
        sep = ', because '
        fields = text.split(sep)
        if len(fields) != 2:
            print("outlier", fields)
        answer, rationale = fields[0], fields[1]
    else:
        raise RuntimeError("Unsupported decoding style")
    
    p["answer"] = answer
    p["rationale"] = rationale
    return p

def analyze_few_shot_post_hoc_pipeline(args):
    dev_set = read_synth_data("data/250-dev_synth.json")
    dev_set = dev_set[:args.num_dev]

    predictions = read_json(result_cache_name(args))
    predictions = [process_joint_prediction(p, args.style) for p in predictions]
    evaluate_joint_predictions(dev_set, predictions, do_print=False)
    print(result_cache_name(args))

if __name__=='__main__':
    args = _parse_args()
    if args.run_prediction:
        test_few_shot_post_hoc_pipeline(args)
    else:
        analyze_few_shot_post_hoc_pipeline(args)
