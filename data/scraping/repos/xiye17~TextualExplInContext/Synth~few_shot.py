import os
import argparse
import openai
from tqdm import tqdm

from utils import *
from dataset_utils import read_synth_data, index_example
from comp_utils import length_of_prompt

openai.api_key = os.getenv("OPENAI_API_KEY")
_MAX_COMP_TOKENS = 25

def _parse_args():
    parser = argparse.ArgumentParser()
    add_engine_argumenet(parser)
    # standard, instruction, etc
    parser.add_argument('--style', type=str, default="standard", choices=["standard"])
    parser.add_argument('--run_prediction', default=False, action='store_true')
    parser.add_argument('--run_length_test', default=False, action='store_true')    
    parser.add_argument('--num_shot', type=int, default=16)
    parser.add_argument('--train_slice', type=int, default=0)
    parser.add_argument('--num_dev', type=int, default=250)
    # parser.add_argument('--strategy', type=str, default="random", choices=["random", "nearest"])

    args = parser.parse_args()
    specify_engine(args)
    return args

def result_cache_name(args):
    return "misc/few_{}_tr{}-{}_dv{}_{}_predictions.json".format(args.engine_name,
                    args.train_slice, args.train_slice + args.num_shot, args.num_dev, args.style)

def random_sample_strategy(ex, training_data, num_shot):
    return training_data[:num_shot]

def nearest_sample_strategy(ex, training_data, num_shot):
    sim_scores = []

    x_words = ex["words"]
    for y in training_data:
        y_words = y["words"]
        score = len(x_words & y_words)
        sim_scores.append(score)

    ranked_examples = sorted(zip(training_data, sim_scores), key=lambda x: x[1], reverse=True)

    return [x[0] for x in ranked_examples[:num_shot]]

def in_context_prediction(ex, shots, engine, style="standard", length_test_only=False):    
    if style == "standard":
        # raise RuntimeError("Unsupported prompt style")
        showcase_examples = [
            "{}\nQ: {}\nA: {}\n".format(s["context"], s["question"], s["answer"]) for s in shots
        ]
        input_example = "{}\nQ: {}\nA:".format(ex["context"], ex["question"])
        
        prompt = "\n".join(showcase_examples + [input_example])
    else:
        raise RuntimeError("Unsupported prompt style")

    if length_test_only:
        pred = length_of_prompt(prompt, _MAX_COMP_TOKENS)
        print("-----------------------------------------")
        print(pred)
        print(prompt)
        return pred
    resp = openai.Completion.create(engine=engine, prompt=prompt, temperature=0.0, max_tokens=_MAX_COMP_TOKENS, logprobs=5, echo=True, stop='\n')

    pred = resp["choices"][0]
    pred["prompt"] = prompt
    pred["text"] = pred["text"][len(prompt):]
    return pred

def normalize_prediction(x):
    fields = x.strip().split()
    if fields:
        return fields[0]
    else:
        return x.strip()

def evaluate_qa_predictions(dev_set, predictions, do_print=False):
    acc = 0
    hit_acc = 0
    for ex, pred in zip(dev_set, predictions):
        gt = ex["answer"]
        orig_p = pred["text"]
        hit_acc += gt in orig_p
        p = normalize_prediction(orig_p)
        acc += gt == p
        if do_print:
            if gt == p:
                print("-----------Correct-----------")            
            else:
                print("-----------Wrong-----------")
            print(pred["prompt"])
            print('PR:', p, "\t|\t", pred["text"])
            print('GT:', gt)

    print("ACC", acc / len(dev_set))
    print("HIT ACC", hit_acc / len(dev_set))

def test_few_shot_performance(args):
    print("Running prediction")

    train_set = read_synth_data("data/100-train_synth.json")    
    dev_set = read_synth_data("data/250-dev_synth.json")

    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]
    dev_set = dev_set[:args.num_dev]

    train_set = [index_example(x) for x in train_set]
    dev_set = [index_example(x) for x in dev_set]
    predictions = []
    for x in tqdm(dev_set, total=len(dev_set), desc="Predicting"):
        predictions.append(in_context_prediction(x, train_set, engine=args.engine, style=args.style, length_test_only=args.run_length_test))

    if args.run_length_test:
        print(result_cache_name(args))
        print('MAX', max(predictions), 'COMP', _MAX_COMP_TOKENS)
        return

    # save
    # read un indexed dev    
    dump_json(predictions, result_cache_name(args))

    # acc
    evaluate_qa_predictions(dev_set, predictions)


def analyze_few_shot_performance(args):
    dev_set = read_synth_data("data/250-dev_synth.json")
    dev_set = dev_set[:args.num_dev]
    predictions = read_json(result_cache_name(args))
    evaluate_qa_predictions(dev_set, predictions, do_print=True)
    print(result_cache_name(args))

if __name__=='__main__':
    args = _parse_args()
    if args.run_prediction:
        test_few_shot_performance(args)
    else:
        analyze_few_shot_performance(args)
