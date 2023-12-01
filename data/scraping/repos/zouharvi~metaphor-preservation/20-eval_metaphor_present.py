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
args.add_argument("-o", "--output", default="data/output_eval/dataset.jsonl")
args = args.parse_args()

openai.api_key_path = "meta/openai_key.txt"

# in case the process gets killed
data = [json.loads(x) for x in open(args.input, "r")]
if os.path.exists(args.output):
    data_out = [json.loads(x) for x in open(args.output, "r")]
    data = data[len(data_out):]
else:
    data_out = []


@backoff.on_exception(
    backoff.expo,
    exception=(openai.error.RateLimitError, openai.error.ServiceUnavailableError, openai.error.APIError),
    max_tries=8, jitter=None
)
def get_metaphor_rating(text):
    time.sleep(0.5)
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful and austere assistant for metaphor detection in text. Reply using only a single number 1 to 5 scale and nothing else."},
            {"role": "user", "content": text},
        ],
        max_tokens=1
    )

    # print(completion)
    rating = completion.choices[0].message.content
    return rating


for line in tqdm.tqdm(data):
    line = copy.deepcopy(line)
    if line["text_lit"]:
        line["text_lit"] = get_metaphor_rating(line["text_lit"])
    if line["text_met"]:
        line["text_met"] = get_metaphor_rating(line["text_met"])

    data_out.append(line)

    # resave everything
    out_file = open(args.output, "w")
    out_file.write("\n".join([
        json.dumps(o, ensure_ascii=False) for o in data_out
    ]) + "\n")


# for MODEL in "paraphrase_bart" "paraphrase_parrot" "paraphrase_paws" "paraphrase_pegasus"; do
# for MODEL in "translate_deepl_cs" "translate_deepl_de" "translate_google_cs" "translate_google_de" "translate_nllb_cs" "translate_nllb_de" "translate_opus_cs" "translate_opus_de"; do
# for MODEL in "translate_deepl_de" "translate_google_de" "translate_opus_de" "translate_nllb_de"; do
#     echo "Running $MODEL";
#     ./src/20-eval_metaphor_present.py -i "data/output/${MODEL}.jsonl" -o "data/eval_present/${MODEL}.jsonl";
#     sleep 10;
# done
