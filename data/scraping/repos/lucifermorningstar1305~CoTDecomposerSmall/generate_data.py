from typing import Any, Dict, Tuple, List, Optional

import cohere
import openai
import os
import sys
import time
import argparse
import json
import logging
import random

from rich.progress import track
from datasets import load_dataset
from collections import defaultdict

from dotenv import load_dotenv
from cot_decomp_few_shot_examples import COTD_FEW_SHOT

random.seed(42)

def filter_yes_no(examples: Dict) -> Dict:
    """
    This function filters out the yes/no questions from the list of 
    questions in HotpotQA to align with the research of CoT Decomposition.

    :param examples: a dictionary 
    """
    if examples["answer"] in ["yes", "no"]:
        return examples
    else:
        return {}


def query_llm(dataset: List[Dict], yes_no: Optional[bool]=False):
    """
    This function will be used to query the LLM with the set of questions

    :param dataset: the dataset to iterate over for CoT decomposition
    :param yes_no: whether the answer is a yes/no type answer.

    :returns: the final curated dataset.
    """

    data_len = len(dataset)
    
    gen_data = defaultdict(lambda: list())
    api_counter = 0

    for i in track(range(data_len)):

        ques = dataset[i]["question"]
        ans = dataset[i]["answer"]

        prompt = COTD_FEW_SHOT + f"\nHuman: Question: {ques}\nChoices:\n(A)yes\n(B)no"

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role":"user",
                 "content": prompt}
            ],
            temperature=.8,
            top_p=.95,
        )

        decomposition = completion.choices[0].message

        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": decomposition["content"]},
                {"role": "user", "content": "\nHuman: Based on the above, what's the correct answer to the question?"}
            ],
            temperature=.8,
            top_p=.95,
        )

        pred_ans = completion.choices[0].message

        logging.info(f"Question: {ques}")
        logging.info(f"Decomposition: {decomposition}")
        logging.info(f"Answer: {pred_ans}")

        gen_data["question"].append(ques)
        gen_data["decomposition"].append(decomposition["content"])
        gen_data["pred_ans"].append(pred_ans["content"])
        gen_data["answer"].append(ans)


    return gen_data


if __name__ == "__main__":
    
    load_dotenv()

    openai.api_key = os.environ["OPENAI_API_KEY"]

    if not os.path.exists("./data/"):
        os.mkdir("./data")

    if not os.path.exists("./logs/"):
        os.mkdir("./logs")

    logging.basicConfig(filename="./logs/generate_data.log", 
                        level=logging.DEBUG,
                        format="%(asctime)s:%(levelname)s:%(name)s:%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, required=True, help="the huggingface dataset name")

    args = parser.parse_args()

    if args.dataset == "hotpot_qa":
        data = load_dataset(args.dataset, 'distractor', split="train")
        new_data = data.filter(filter_yes_no)
        new_data = new_data.shuffle(seed=42).select(range(10_000))
        generated_dataset = query_llm(new_data, yes_no=True)

        with open("data/"+args.dataset+".json", "w") as fp:
            json.dump(generated_dataset, fp)
    else:
        data = load_dataset(args.dataset, split="train")
