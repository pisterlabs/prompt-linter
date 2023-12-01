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
                    # {"index": "dev-0", "cause": "The woman gave birth to a child.", "effect": "The child brought psycho-physical phenomena on a new life.", "conceptual_explanation": "Birth is the arising of the psycho-physical phenomena."}

                    prompt = """Cause: %s
Effect: %s
Question: why the cause can lead to the effect ?
Answer:""" % (item["cause"], item["effect"])
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
                    fout.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == '__main__':
    """
    set "input_file" and "gpt_type" to control select dataset and ChatGPT's version 

    set "openai_api_key" as your own API key.


    For example:

    python CEG.py \
    --input_file data/e-CARE/e-CARE-main-code/dataset/Explanation_Generation/dev.jsonl \
    --output_dir output/CEG/e-CARE/dev \
    --gpt_type text-davinci-003 \
    --openai_api_key YOUR_API_KEY
    """
    parser = ArgumentParser()
    parser.add_argument('--input_file', type=str, default="data/e-CARE/e-CARE-main-code/dataset/Explanation_Generation/dev.jsonl")
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--gpt_type', type=str, choices=["text-davinci-003", "gpt-3.5-turbo", "gpt-4"])
    parser.add_argument('--openai_api_key', type=str)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    process(args)
