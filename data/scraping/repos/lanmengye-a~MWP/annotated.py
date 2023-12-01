import json

import openai
from typing import Dict, Any
from datetime import datetime
import argparse




def create_reader_request(example: Dict[str, Any]) -> str:
    if not example["Body"].endswith('.'):
        example["Body"] += '.'
    string = f'{example["Body"]} {example["Question"]}'
    return string


from tools.utilities1 import get_gpt3_output
import re
def get_rationales( question,few_prompt=None,answer=None):
    # openai.api_key = args.key
    ques = re.split("\.|\,",question)[-1]
    body = re.split("\.|\,",question)[:-1]
    body = ".".join(body)
    prefix = """
# Read the following passages to answer questions with Python code,store the result as a 'ans' variable:
"""
    midfix = f"""
# Passage: {body}
# Question:{ques}
# Thought:
# Let's think step by step to derive the answer, store the result as a 'ans' variable:
import numpy as np
# According to the question, we can define the variables:
"""
    if few_prompt is not None:
        message = prefix.strip("\n") + "\n\n" + few_prompt.strip() + "\n" + midfix + "\n"
    else:
        message = prefix.strip("\n") + "\n\n" + midfix.strip("\n") + "\n"
    program ,prediction = get_gpt3_output(message,patience = 5,midfix=midfix,label = answer)
    return prediction, program


if __name__ == "__main__":
    now = datetime.now()
    dt_string = now.strftime("%m_%d_%H_%M")
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", default='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', type=str)
    parser.add_argument("--dry_run", default=False, action='store_true')
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--end", default=-1, type=int)
    args = parser.parse_args()
    with open("dataset/demos/svamp/demo8_train.json","r") as reader:
        lines = json.load(reader)
    lines = lines["demo"]
    newlines = []
    for line in lines:
        ques = line["question"]
        ans = line["gold_ans"]
        prediction,program = get_rationales(args,ques,ans)
        newlines.append({"Question":ques,
                         "Thought": program,
                         "Prediction":prediction,
                         "Answer":ans,
                         "index":line["index"]
                         })
    with open("dataset/demos/svamp/0814demo8_train_annotated_5.json", "w") as writer:
        json.dump(newlines,writer,ensure_ascii=False,indent=4)