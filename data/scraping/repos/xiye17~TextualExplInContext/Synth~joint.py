import os
import argparse
import openai
from tqdm import tqdm

from utils import *
from dataset_utils import read_synth_data, index_example, reorder_rationale
from comp_utils import length_of_prompt
from sklearn.metrics import confusion_matrix

openai.api_key = os.getenv("OPENAI_API_KEY")
_MAX_COMP_TOKENS = 48

def _parse_args():
    parser = argparse.ArgumentParser()
    add_engine_argumenet(parser)

    # standard, instruction, etc
    parser.add_argument('--style', type=str, default="e-p")
    parser.add_argument('--run_prediction', default=False, action='store_true')
    parser.add_argument('--run_length_test', default=False, action='store_true')    
    parser.add_argument('--num_shot', type=int, default=16)
    parser.add_argument('--train_slice', type=int, default=0)
    parser.add_argument('--num_dev', type=int, default=250)
    parser.add_argument('--reorder', default=False, action='store_true')

    args = parser.parse_args()
    specify_engine(args)
    return args

def result_cache_name(args):
    return "misc/joint_{}_tr{}-{}_dv{}_{}_predictions.json".format(args.engine_name,
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

def proc_reversed_rationale(text):
    text = text[:-1]
    a0, a1 = text.split(" and ")    
    return a1 + " and " + a0

# return prompt stop_signal
def prompt_for_joint_prediction(ex, shots, style):
    stop_signal = "\n\n"
    if style == "p-e-r": # P-E Reversed
        showcase_examples = [
            "{}\nQ: {}\nA: {}, because {}\n".format(s["context"], s["question"], s["answer"],  s["text_rationale"]) for s in shots
        ]
        input_example = "{}\nQ: {}\nA:".format(ex["context"], ex["question"])

        prompt = "\n".join(showcase_examples + [input_example])
    elif style == "e-p-r": # E-P Reversed
        showcase_examples = [
            "{}\nQ: {}\nA: Because {} the answer is {}.\n".format(s["context"], s["question"], s["text_rationale"][:-1] + ',',  s["answer"]) for s in shots
        ]
        input_example = "{}\nQ: {}\nA:".format(ex["context"], ex["question"])

        prompt = "\n".join(showcase_examples + [input_example])
    elif style == "e-p": # E-P
        showcase_examples = [
            "{}\nQ: {}\nA: Because {} the answer is {}.\n".format(s["context"], s["question"], proc_reversed_rationale(s["text_rationale"]) + ',',  s["answer"]) for s in shots
        ]
        input_example = "{}\nQ: {}\nA:".format(ex["context"], ex["question"])

        prompt = "\n".join(showcase_examples + [input_example])
    elif style == "p-e": # P-E
        showcase_examples = [
            "{}\nQ: {}\nA: {}, because {}.\n".format(s["context"], s["question"], s["answer"], proc_reversed_rationale(s["text_rationale"])) for s in shots
        ]
        input_example = "{}\nQ: {}\nA:".format(ex["context"], ex["question"])

        prompt = "\n".join(showcase_examples + [input_example])
    else:
        raise RuntimeError("Unsupported prompt style")
    return prompt, stop_signal    

def in_context_joint_prediction(ex, shots, engine, style="standard", length_test_only=False):

    prompt, stop_signal = prompt_for_joint_prediction(ex, shots, style)
    if length_test_only:
        pred = length_of_prompt(prompt, _MAX_COMP_TOKENS)
        print("-----------------------------------------")
        print(pred)
        print(prompt)
        return pred
    resp = openai.Completion.create(engine=engine, prompt=prompt, temperature=0.0, max_tokens=_MAX_COMP_TOKENS, echo=True, logprobs=5, stop=stop_signal)    
    pred = resp["choices"][0]
    pred["prompt"] = prompt
    pred["text"] = pred["text"][len(prompt):]
    pred["completion_offset"] = len(prompt)
    return pred

def is_factual(rat, context):
    for x in rat.split(' and '):
        x = x.lstrip()
        x = x.rstrip('.')
        x = x.rstrip(',')
        if x not in context:
            return False
    return True


def evaluate_joint_predictions(dev_set, predictions, do_print=False):
    qa_acc = 0
    set_acc = 0
    acc_records = []
    fact_records = []
    cons_records = []
    
    for ex, pred in zip(dev_set, predictions):
        ans_gt = ex["answer"]
        rationale_gt = ex["text_rationale"]
        rationale_p = pred["rationale"]
        ans_p = pred["answer"]

        matched = evaluate_rationale_match(rationale_gt, rationale_p)
        set_acc += matched
        qa_acc += ans_gt == ans_p
        # consistency = rationale_p.split(" ")[0] == ans_p
        if " and " in rationale_p:
            pre_consistency = rationale_p.split(" and ")[1].split(" ")[0] == ans_p
            toks = rationale_p.rstrip(',').rstrip('.').split(" ")
            consistency = pre_consistency and (toks[0] == toks[-1])
            factuality = is_factual(rationale_p, ex["context"])
        else:
            factuality = is_factual(rationale_p, ex["context"])
            consistency = False
        acc_records.append(ans_gt == ans_p)
        fact_records.append(factuality)
        cons_records.append(consistency)

        if do_print:
            if ans_p == ans_gt:
                # continue
                print("-----------Correct-----------")            
            else:
                print("-----------Wrong-----------")
            print(pred["prompt"].split("\n\n")[-1])
            # print('Completion:', pred["text"].strip())
            print("Acc:", ans_gt == ans_p, "Cons:", consistency, "Fact:", factuality)
            print('PR:', ans_p, '\t|\t', rationale_p)
            print('GT:', ans_gt,  '\t|\t', rationale_gt)

    print("QA ACC", qa_acc / len(dev_set))
    print("Rationale Set ACC", set_acc / len(dev_set))
    print("Cons ACC", sum(cons_records) / len(dev_set))
    print("Fact ACC", sum(fact_records) / len(dev_set))
    print("ACC==CONS", sum([a == b for (a,b) in zip(acc_records, cons_records)])/250)
    print("ACC==FACT",sum([a == b for (a,b) in zip(acc_records, fact_records)])/250)
    print("Conf Cons", confusion_matrix(acc_records,cons_records)/250)
    print("Conf Fact", confusion_matrix(acc_records,fact_records)/250)

def conditional_strip_prompt_prefix(x, p):
    if x.startswith(p):
        x = x[len(p):]
    return x.strip()

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
    elif style == "e-p-r":
        sep = ' the answer is '
        fields = text.split(sep)
        if len(fields) != 2:
            print("outlier", fields)
        answer, rationale = fields[1], fields[0]
        answer = answer.rstrip('.')
        rationale = rationale[len("Because "):]
    elif style == "e-p":
        sep = ' the answer is '
        fields = text.split(sep)
        if len(fields) != 2:
            print("outlier", fields)
        answer, rationale = fields[1], fields[0]
        answer = answer.rstrip('.')
        rationale = rationale[len("Because "):]
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

def test_few_shot_joint_prediction(args):
    print("Running prediction")
    train_set = read_synth_data("data/100-train_synth.json")
    dev_set = read_synth_data("data/250-dev_synth.json")

    train_set = train_set[args.train_slice:(args.train_slice + args.num_shot)]
    dev_set = dev_set[:args.num_dev]


    train_set = [index_example(x) for x in train_set]
    dev_set = [index_example(x) for x in dev_set]
    predictions = []
    for x in tqdm(dev_set, total=len(dev_set), desc="Predicting"):
        predictions.append(in_context_joint_prediction(x, train_set, engine=args.engine, style=args.style, length_test_only=args.run_length_test))

    if args.run_length_test:
        print(result_cache_name(args))
        print('MAX', max(predictions), 'COMP', _MAX_COMP_TOKENS)
        return

    # print(predictions)
    # save
    # read un indexed dev
    dump_json(predictions, result_cache_name(args))
    predictions = [process_joint_prediction(p, args.style) for p in predictions]
    # acc
    evaluate_joint_predictions(dev_set, predictions)

def evaluate_rationale_match(gt, p):
    if gt == p:
        return True
    if "." not in p:
        return False
    sent_boundary = p.index('.')
    fst_sent = p[:(sent_boundary + 1)]
    snd_sent = p[(sent_boundary + 2):]

    return (snd_sent + ' ' + fst_sent) == gt


def analyze_few_shot_rationalization(args):
    dev_set = read_synth_data("data/250-dev_synth.json")
    dev_set = dev_set[:args.num_dev]

    predictions = read_json(result_cache_name(args))
    predictions = [process_joint_prediction(p, args.style) for p in predictions]
    evaluate_joint_predictions(dev_set, predictions, do_print=True)
    print(result_cache_name(args))

if __name__=='__main__':
    args = _parse_args()
    if args.run_prediction:
        test_few_shot_joint_prediction(args)
    else:
        analyze_few_shot_rationalization(args)
