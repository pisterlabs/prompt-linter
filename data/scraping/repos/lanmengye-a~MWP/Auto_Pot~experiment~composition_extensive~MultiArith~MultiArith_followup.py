'''
Adapted from https://github.com/kojima-takeshi188/zero_shot_cot
'''
from tool import *
import argparse
from time import sleep
import openai
from chatgpt0 import call_gpt3
from collections import Counter
import collections
def get_rationales(args, question):
    openai.api_key = args.key
    question = question.strip().strip("# Question:")
    result_counter = Counter()
    result_dict = collections.defaultdict(list)
    practice = 3
    codes = []
    while practice!=0:
        try:
            program = call_gpt3("text-davinci-003", question)
            codes.append(program)
            ans = safe_execute(program)
            ans = floatify_ans(ans)
            practice -= 1
        except Exception as e:
            print(e)
            sleep(3)
            practice -= 1
        if ans is not None and not isinstance(ans, str):
            result_counter.update([abs(ans)])
            result_dict[ans].append(abs(practice-3))
    if len(result_counter) > 0:
        prediction = result_counter.most_common(1)[0][0]
    else:
        prediction = None
    if prediction is not None:
        try:
            program = codes[result_dict[prediction][0]].replace("    ", "")
        except Exception as e:
            print("result_dict:",codes)
    if program is not None:
        print("program:", program)
    if ans is not None:
        print("prediction:", prediction)
    return program,prediction
import json
def main():
    args = parse_arguments()
    print('*****************************')
    print(args)
    print('*****************************')
    with open('MultiArith.jsonl', 'r') as f:
        lines = f.readlines()
        lines = [json.loads(line) for line in lines]

    with open("multiArith_out.jsonl","a+") as writer:
        for idx,item in enumerate(lines):
            if idx==100:
                break
            question = item['question']
            program, ans = get_rationales(args, question)
            item['program'] = program
            item['ans'] = ans
            writer.write(json.dumps({
                "question":item['question'],
                "program":item['program'],
                "gold_ans": item['gold_ans'],
                "pred_ans":item['ans']})+"\n"
            )


def parse_arguments():
    parser = argparse.ArgumentParser(description="Zero-shot-CoT")

    parser.add_argument("--random_seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--dataset", type=str, default="svamp",
        choices=["aqua", "gsm8k", "commonsensqa", "addsub", "multiarith", "strategyqa", "svamp", "singleeq",
                 "coin_flip", "last_letters"], help="dataset used for experiment"
    )
    parser.add_argument(
        "--demo_path", type=str, default="demos/svamp_pot_optim_3", help="pre-generated demos used for experiment"
    )
    parser.add_argument(
        "--resume_id", type=int, default=0,
        help="resume from which question id (current line number in the output file), if the experiment fails accidently (e.g., network error)"
    )
    parser.add_argument("--minibatch_size", type=int, default=1, choices=[1],
                        help="minibatch size should be 1 because GPT-3 API takes only 1 input for each request")

    parser.add_argument("--max_num_worker", type=int, default=0, help="maximum number of workers for dataloader")

    parser.add_argument(
        "--model", type=str, default="gpt3-xl",
        choices=["gpt3", "gpt3-medium", "gpt3-large", "gpt3-xl", "code-davinci-002"],
        help="model used for decoding. Note that 'gpt3' are the smallest models."
    )
    parser.add_argument(
        "--dry_run", type=bool, default=False, help="debug mode"
    )
    parser.add_argument(
        "--key", type=str, default="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"

    )
    parser.add_argument(
        "--method", type=str, default="auto_cot",
        choices=["zero_shot", "zero_shot_cot", "few_shot", "few_shot_cot", "auto_cot"], help="method"
    )
    parser.add_argument(
        "--output_dir", type=str, default="experiment/svamp1000.jsonl", help="output directory"
    )
    parser.add_argument(
        "--max_length_cot", type=int, default=256,
        help="maximum length of output tokens by model for reasoning extraction"
    )
    parser.add_argument(
        "--max_length_direct", type=int, default=32,
        help="maximum length of output tokens by model for answer extraction"
    )
    parser.add_argument(
        "--limit_dataset_size", type=int, default=0,
        help="whether to limit test dataset size. if 0, the dataset size is unlimited and we use all the samples in the dataset for testing."
    )
    parser.add_argument(
        "--api_time_interval", type=float, default=1.0,
        help="sleep between runs to avoid excedding the rate limit of openai api"
    )
    parser.add_argument(
        "--temperature", type=float, default=0, help="temperature for GPT-3"
    )
    parser.add_argument(
        "--log_dir", type=str, default="./log/", help="log directory"
    )

    args = parser.parse_args()

    if args.dataset == "aqua":
        args.dataset_path = "./dataset/AQuA/test_raw.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
    elif args.dataset == "gsm8k":
        args.dataset_path = "./dataset/grade-school-math/test_raw.jsonl"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "commonsensqa":
        args.dataset_path = "./dataset/CommonsenseQA/dev_rand_split.jsonl"
        args.direct_answer_trigger = "\nTherefore, among A through E, the answer is"
        args.plausible_answer_trigger = "Choose the most plausible answer from among choices A through E."
    elif args.dataset == "addsub":
        args.dataset_path = "./dataset/AddSub/AddSub.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "multiarith":
        args.dataset_path = "./dataset/MultiArith/MultiArith.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "strategyqa":
        args.dataset_path = "./dataset/StrategyQA/task.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "svamp":
        args.dataset_path = "./dataset/SVAMP/SVAMP.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "singleeq":
        args.dataset_path = "./dataset/SingleEq/questions.json"
        args.direct_answer_trigger = "\nTherefore, the answer (arabic numerals) is"
    elif args.dataset == "bigbench_date":
        args.dataset_path = "./dataset/Bigbench_Date/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through F, the answer is"
    elif args.dataset == "object_tracking":
        args.dataset_path = "./dataset/Bigbench_object_tracking/task.json"
        args.direct_answer_trigger = "\nTherefore, among A through C, the answer is"
    elif args.dataset == "coin_flip":
        args.dataset_path = "./dataset/coin_flip/coin_flip.json"
        args.direct_answer_trigger = "\nTherefore, the answer (Yes or No) is"
    elif args.dataset == "last_letters":
        args.dataset_path = "./dataset/last_letters/last_letters.json"
        args.direct_answer_trigger = "\nTherefore, the answer is"
    else:
        raise ValueError("dataset is not properly defined ...")

    # "Therefore, the answer ..." -> "The answer ..."
    trigger = args.direct_answer_trigger.replace("\nTherefore, ", "")
    args.direct_answer_trigger_for_zeroshot = trigger[0].upper() + trigger[1:]
    args.direct_answer_trigger_for_zeroshot_cot = args.direct_answer_trigger
    args.direct_answer_trigger_for_fewshot = "The answer is"
    args.cot_trigger = "Let's think step by step."

    return args


if __name__ == "__main__":
    main()