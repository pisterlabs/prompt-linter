#!/usr/bin/env python3

import openai
import os
import json
import copy
import time
import tqdm
import backoff
import argparse

args = argparse.ArgumentParser()
args.add_argument("-i", "--input", default="data/output/dataset.jsonl")
args.add_argument("-o", "--output", default="data/eval_preserved/dataset.jsonl")
args = args.parse_args()

openai.api_key_path = "meta/openai_key.txt"

# in case the process gets killed
data_new = [json.loads(x) for x in open(args.input, "r")]
data_src = [json.loads(x) for x in open("data/output/dataset.jsonl", "r")]
if os.path.exists(args.output):
    data_out = [json.loads(x) for x in open(args.output, "r")]
    data_new = data_new[len(data_out):]
    data_src = data_src[len(data_out):]
else:
    data_out = []


@backoff.on_exception(
    backoff.expo,
    exception=(openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError),
    max_tries=16, jitter=None
)
def get_metaphor_rating(text_src, text_new):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful and austere assistant for detecting how much is the true meaning preserved. Reply using only a single number 1 (not at all) to 5 (completely, including style) and nothing else."},
            {"role": "user", "content": "Source: " + text_src},
            {"role": "user", "content": "Hypothesis: " + text_new},
        ],
        max_tokens=1
    )

    # print(completion)
    rating = completion.choices[0].message.content
    return rating


for line_src, line_new in tqdm.tqdm(zip(data_src, data_new), total=len(data_src)):
    line_src = copy.deepcopy(line_src)
    if line_src["text_lit"]:
        line_src["text_lit"] = get_metaphor_rating(line_src["text_lit"], line_new["text_lit"])
    if line_src["text_met"]:
        line_src["text_met"] = get_metaphor_rating(line_src["text_met"], line_new["text_met"])

    data_out.append(line_src)

    # resave everything
    out_file = open(args.output, "w")
    out_file.write("\n".join([
        json.dumps(o, ensure_ascii=False) for o in data_out
    ]) + "\n")


# for MODEL in "paraphrase_bart" "paraphrase_parrot" "paraphrase_paws" "paraphrase_pegasus" "translate_deepl_cs" "translate_deepl_de" "translate_google_cs" "translate_google_de" "translate_nllb_cs" "translate_nllb_de" "translate_opus_cs" "translate_opus_de"; do
# for MODEL in "translate_deepl_de" "translate_google_de" "translate_nllb_de" "translate_opus_de"; do
#     echo "Running $MODEL";
#     ./src/21-eval_metaphor_preserved.py -i "data/output/${MODEL}.jsonl" -o "data/eval_preserved/${MODEL}.jsonl";
#     sleep 10;
# done
