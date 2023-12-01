# -*- coding: utf-8 -*-

import json
import os
import shutil
import traceback
from argparse import ArgumentParser

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

from utils.log_kits import get_logger

from utils.build_ICL_data import get_e_CARE_task3_demonstrations, get_task3_COT

@retry(wait=wait_random_exponential(min=10, max=20), stop=stop_after_attempt(10))
def completion_with_backoff_davinci(args, logger, **kwargs):
    try:
        response = openai.Completion.create(**kwargs)
        response_ls = [response["choices"][i]["text"] for i in range(len(kwargs["prompt"]))]
        return response_ls
    except Exception as e:
        logger.warning("\n\n" + traceback.format_exc() + "\n\n")
        raise e

@retry(wait=wait_random_exponential(min=10, max=20), stop=stop_after_attempt(10))
def completion_with_backoff_chat(args, logger, **kwargs):
    try:
        response = openai.ChatCompletion.create(**kwargs)
        return response
    except Exception as e:
        logger.warning("\n\n" + traceback.format_exc() + "\n\n")
        raise e


def process(args):
    openai.api_key = args.openai_api_key

    logger = get_logger(os.path.join(args.output_dir, "log"), is_console=True)

    logger.info(str(args))

    for key, value in args.__dict__.items():
        logger.info("%s :\t%s" % (key, str(value)))

    ICL_str = ""
    if args.ICL == "zero-shot-COT":
        ...
    elif args.ICL != "None":
        if args.ICL.startswith("COT_"):
            icl_method = get_task3_COT
            # COT_2:3
            pos_num, neg_num = [int(t) for t in args.ICL[4:].split(":")]
        else:
            # 2:3
            icl_method = get_e_CARE_task3_demonstrations
            pos_num, neg_num = [int(t) for t in args.ICL.split(":")]

        ICL_list = icl_method(pos_num, neg_num)
        ICL_str = "\n\n".join(ICL_list) + "\n\n"


    with open(args.input_file, 'r', encoding="utf-8") as fin:
        with open(os.path.join(args.output_dir, "response.json"), 'w+', encoding="utf-8") as fout:
            item_list = [json.loads(line) for line in fin.readlines()]
            batch_size = 2
            item_buckets = [item_list[t:t + batch_size] for t in range(0, len(item_list), batch_size)]

            # total ? item_buckets
            for bucket_no, item_buck in enumerate(item_buckets):
                if args.debug and bucket_no == 2:
                    break
                logger.info("#" * 10 + "bucket " + str(bucket_no + 1) + "#" * 10)

                prompts = []

                for item in item_buck:
                    # {"index": "dev-0", "premise": "Thife.", "ask-for": "cause", "hypothesis1": "Thed.", "hypothesis2": "The ations.", "label": 0}

                    if args.ICL == "zero-shot-COT":
                        base_p = """Event A: %s
Event B: %s
Question: is there a causal relationship between Event A and Event B ? Let's think step by step.
Answer:"""
                    elif not args.ICL.startswith("COT_"):
                        base_p = """Event A: %s
Event B: %s
Question: is there a causal relationship between Event A and Event B ?
Answer:"""
                    else:
                        base_p = """Event A: %s
Event B: %s
Question: is there a causal relationship between Event A and Event B ?
Reasoning Process:"""

                    prompt = ICL_str + base_p % (item["premise"], item["hypothesis1"])
                    prompts.append(prompt)
                    prompt = ICL_str + base_p % (item["premise"], item["hypothesis2"])
                    prompts.append(prompt)

                if args.gpt_type in ("text-davinci-002", "text-davinci-003"):
                    response_ls = completion_with_backoff_davinci(args, logger,
                                                                  model=args.gpt_type,
                                                                  prompt=prompts,
                                                                  temperature=0,
                                                                  max_tokens=1024,
                                                                  )
                elif args.gpt_type in ("gpt-3.5-turbo", "gpt-4"):
                    response_ls = []
                    for prompt in prompts:
                        response = completion_with_backoff_chat(args, logger,
                                                           model=args.gpt_type,
                                                           messages=[
                                                               {"role": "user", "content": prompt}
                                                           ],
                                                           temperature=0,
                                                           max_tokens=1024,
                                                           )
                        response_ls.append(response["choices"][0]["message"]["content"])
                else:
                    assert False


                assert len(response_ls) == len(prompts)
                # feed responses to json
                for tmp_idx, (response, prompt) in enumerate(zip(response_ls, prompts)):
                    item = item_buck[tmp_idx // 2]
                    logger.info("\nPrompt+response:\n" + prompt + response)
                    if tmp_idx % 2 == 0:
                        item["gpt_response_1"] = response
                    elif tmp_idx % 2 == 1:
                        item["gpt_response_2"] = response
                        fout.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    """
    set "input_file" and "gpt_type" to control select dataset and ChatGPT's version 
    
    set "openai_api_key" as your own API key.
    
    set "ICL" as:
    1. "None": perform zero-shot ChatGPT
    2. "x:y": perform ICL with x positive and y negative demonstrations, such as "2:3".
    3. "zero-shot-COT": perform ChatGPT with zero-shot COT, i.e, add "Let's think step by step." into the prompt.
    4. "COT_x:y": perform ChatGPT with COT, with x positive and y negative demonstrations, such as "COT_4:8".
    
    For example:
    
    python CD_binary_classification.py \
    --input_file data/COPA/copa.all.jsonl \
    --output_dir output/COPA/ChatGPT-3.5-turbo/zero-shot \
    --gpt_type gpt-3.5-turbo \
    --openai_api_key xxx \
    --ICL zero-shot-COT
    """
    parser = ArgumentParser()
    parser.add_argument('--input_file', type=str,choices=["data/COPA/copa.all.jsonl", "data/e-CARE/e-CARE-main-code/dataset/Causal_Reasoning/dev.jsonl"])
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--gpt_type', type=str, choices=["text-davinci-002", "text-davinci-003", "gpt-3.5-turbo", "gpt-4"])
    parser.add_argument('--openai_api_key', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ICL', type=str,default="None")


    args = parser.parse_args()
    process(args)

