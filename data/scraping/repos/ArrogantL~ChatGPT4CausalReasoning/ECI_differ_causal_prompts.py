# -*- coding: utf-8 -*-

import json
import os
import re
import traceback
from argparse import ArgumentParser

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

from utils.log_kits import get_logger


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
                    words = item["words"]
                    sent = " ".join(words)
                    events = item["events"]
                    event1 = " ".join([words[t] for t in events[0]])
                    event2 = " ".join([words[t] for t in events[1]])
                    if args.prompt_style == "one_step":
                        base_prompt = """Input: %s
Question: is there a one-step causal relationship between \"%s\" and \"%s\" ?
Answer:"""
                    elif args.prompt_style == "counterfactual":
                        base_prompt = """Input: %s
Question: if \"%s\" does not happen, will \"%s\" happen ?
Answer:"""
                    else:
                        assert args.prompt_style.startswith("trigger_")
                        trigger_str = " ".join(args.prompt_style.split("_")[1:])
                        base_prompt = "Input: %s\nQuestion: does \"%s\" " + trigger_str + " \"%s\" ?\nAnswer:"

                    prompt = base_prompt % (sent, event1, event2)
                    prompts.append(prompt)

                    prompt = base_prompt % (sent, event2, event1)
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

                        item["gpt_pred_bi_causal_label"] = [0, 0]

                        for tmp_strid, tmp_str in enumerate((item["gpt_response_1"], item["gpt_response_2"])):
                            tmp_words = re.split("[^a-z]", tmp_str.strip().lower())
                            for w in tmp_words:
                                if args.prompt_style == "counterfactual":
                                    if w == "no":
                                        item["gpt_pred_bi_causal_label"][tmp_strid] = 1
                                        break
                                else:
                                    if w == "yes":
                                        item["gpt_pred_bi_causal_label"][tmp_strid] = 1
                                        break

                        fout.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    """
    set "input_file" and "gpt_type" to control select dataset and ChatGPT's version 
    
    set "openai_api_key" as your own API key.
    
    set "prompt_style" as:
    1. "one_step" for one-step prompt
    2. "counterfactual" for counterfactual prompt
    3. "trigger_" + trigger_word for trigger prompt, such as "trigger_lead_to", "trigger_result_in"
    
    For example,
    python ECI_differ_causal_prompts.py \
        --input_file data/CTB/causal_time_bank.json \
        --output_dir output/CTB/triggers/lead_to \
        --gpt_type text-davinci-002 \
        --openai_api_key YOUR_API_KEY \
        --prompt_style trigger_lead_to
    
    """
    parser = ArgumentParser()
    parser.add_argument('--input_file', type=str, choices=["data/CTB/causal_time_bank.json", "data/ESC/intra_sent_causality.json",
                                                           "data/MAVEN-ERE/all.with_neg.ECI.with_docname_sentid.valid.json"])
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--gpt_type', type=str, choices=["text-davinci-002", "text-davinci-003", "gpt-3.5-turbo", "gpt-4"])
    parser.add_argument('--openai_api_key', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--prompt_style', type=str)

    args = parser.parse_args()
    process(args)
