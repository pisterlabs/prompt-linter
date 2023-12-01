# -*- coding: utf-8 -*-

import json
import os
import re
import shutil
import traceback
from argparse import ArgumentParser

import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

from utils.build_ICL_data import get_ESC_demonstrations,get_esc_COT

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

    ICL_str = ""
    if args.ICL == "zero-shot-COT":
        ...
    elif args.ICL != "None":
        if args.ICL.startswith("COT_"):
            icl_method = get_esc_COT
            # COT_2:3
            pos_num, neg_num = [int(t) for t in args.ICL[4:].split(":")]
        else:
            # 2:3
            icl_method = get_ESC_demonstrations
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
                    words=item["words"]
                    sent = " ".join(words)
                    events=item["events"]
                    event1 = " ".join([words[t] for t in events[0]])
                    event2 = " ".join([words[t] for t in events[1]])

                    if args.ICL == "zero-shot-COT":
                        prompt = """Input: %s
Question: is there a causal relationship between \"%s\" and \"%s\" ? Let's think step by step.
Answer:""" % (sent, event1, event2)
                    elif not args.ICL.startswith("COT_"):
                        prompt = ICL_str + """Input: %s
Question: is there a causal relationship between \"%s\" and \"%s\" ?
Answer:""" % (sent, event1, event2)
                    else:
                        prompt = ICL_str + """Input: %s
Question: is there a causal relationship between \"%s\" and \"%s\" ?
Reasoning Process:""" % (sent, event1, event2)

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
                for item, response, prompt in zip(item_buck, response_ls, prompts):
                    logger.info("\nPrompt+response:\n" + prompt + response)
                    item["gpt_response"] = response


                    try:
                        tmp_str=response.split("Answer:",maxsplit=1)[-1]
                    except:
                        tmp_str =response

                    gpt_pred_bi_causal_label=0
                    tmp_words=re.split("[^a-z]",tmp_str.strip().lower())
                    for w in tmp_words:
                        if w == "yes":
                            gpt_pred_bi_causal_label=1
                            break
                        elif w == "no":
                            gpt_pred_bi_causal_label=0
                            break
                    item["gpt_pred_bi_causal_label"]=gpt_pred_bi_causal_label

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
    
    python ECI.py \
    --input_file data/CTB/causal_time_bank.json \
    --output_dir output/CTB \
    --gpt_type text-davinci-002 \
    --openai_api_key YOUR_API_KEY \
    --ICL COT_4:2
    
    
    
    """
    parser = ArgumentParser()
    parser.add_argument('--input_file', type=str,choices=["data/CTB/causal_time_bank.json", "data/ESC/intra_sent_causality.json","data/MAVEN-ERE/all.with_neg.ECI.with_docname_sentid.valid.json"])
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--gpt_type', type=str, choices=["text-davinci-002", "text-davinci-003", "gpt-3.5-turbo", "gpt-4"])
    parser.add_argument('--openai_api_key', type=str)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--ICL', type=str,default="None")

    args = parser.parse_args()
    process(args)



