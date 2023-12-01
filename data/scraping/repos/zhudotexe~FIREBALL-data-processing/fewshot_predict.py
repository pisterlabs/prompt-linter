import glob
import json
import logging
import pathlib
import os
from time import sleep

import tqdm.contrib.logging

import prompts
from dataset.utils import read_jsonl_file
import openai
openai.organization = "org-bgAXfs8WdU5942SLngg0OGpd"
openai.api_key = os.getenv("OPENAI_API_KEY")
NORMALIZED_IN_DIR = pathlib.Path("extract/")
PROMPT_DIR = pathlib.Path("few-shot/prompts")
NUM_SHOTS = 3
OUT_DIR = pathlib.Path("few-shot")
FT_MODEL = 'code-davinci-002'
FT_MODEL_STR = 'code'
# FT_MODEL = 'text-davinci-002'
# FT_MODEL_STR = "text"

text_file = open(PROMPT_DIR / pathlib.Path(f'utt-cmd-full-{NUM_SHOTS}.txt'), "r")
fewshot_prompt_full = text_file.read()
text_file.close()
text_file = open(PROMPT_DIR / pathlib.Path(f'utt-cmd-nostate-{NUM_SHOTS}.txt'), "r")
fewshot_prompt_nostate = text_file.read()
text_file.close()

def writeline(f, d):
    f.write(json.dumps(d))
    f.write("\n")
def main():
    test_file = NORMALIZED_IN_DIR / pathlib.Path('ft-utt-cmd-test-1000.jsonl')
    test_stream = read_jsonl_file(test_file)
    predictions = []
    for data in tqdm.tqdm(test_stream):
        prompt_full = fewshot_prompt_full + prompts.utt_cmd_prompt(data)
        completion_full = openai.Completion.create(
            model=FT_MODEL,
            prompt=prompt_full,
            temperature=0.2,
            max_tokens=128,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n<|aeot|>"],
        )
        data['prediction_full'] = completion_full['choices'][0]['text']
        predictions.append(data)
        prompt_nostate = fewshot_prompt_nostate + prompts.utt_cmd_prompt(data, ablations=["actors", "current"])
        completion_nostate = openai.Completion.create(
            model=FT_MODEL,
            prompt=prompt_nostate,
            temperature=0.2,
            max_tokens=128,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=["\n<|aeot|>"],
        )
        data['prediction_nostate'] = completion_nostate['choices'][0]['text']
        if FT_MODEL_STR == 'code': sleep(6) # rate limit for codex


        

    testf = open(OUT_DIR / f"utt-cmd-{FT_MODEL_STR}-{NUM_SHOTS}-shot-test-predictions.jsonl", mode="w")
    for prompt in predictions:
        writeline(testf, prompt)
    testf.close()
    # testf = open(OUT_DIR / f"utt-cmd-fewshot-test-predictions-nostate.jsonl", mode="w")
    # for prompt in predictions_nostate:
    #     writeline(testf, prompt)
    # testf.close()    

if __name__ == "__main__":
    main()