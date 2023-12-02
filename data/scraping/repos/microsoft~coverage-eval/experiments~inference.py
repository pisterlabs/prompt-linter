import sys
import openai
import json
from pathlib import Path
from random import sample
import backoff
from openai.error import RateLimitError
import anthropic


import ray
ray.init()


system = """You are a terminal.

Instruction:

When use runs:

`coverage run -m  pytest code.py`

then you'll cat the file `code.py`, with each line starting with either of the two symbols below:

> if the line is executed
! is the line is not executed


Example output:

> line1
! line2
> line3
...
> linen

You job is to figure out which line will be executed given different test cases.
"""

@backoff.on_exception(backoff.expo, RateLimitError)
def bard(prompt, system, api_key=None):
    import google.generativeai as palm
    palm.configure(api_key=api_key)
    completion = palm.generate_text(
        model="models/text-bison-001",
        prompt=system + "\n\n" + prompt,
        temperature=0,
        top_p=0,
        # The maximum length of the response
        max_output_tokens=2000,
        stop_sequences=['(anaconda3-2020.11)', 'def test()'],
    )
    return completion.result



@backoff.on_exception(backoff.expo, RateLimitError)
def gpt(prompt, system, model="gpt4", api_key=None):
    openai.api_key = api_key

    messages = [{"role": "system", "content": system}, {"role": "user", "content": prompt}]

    if model == "gpt4":
        engine="gpt-4"
    elif model == "gpt3":
        engine="gpt-3.5-turbo"
    else:
        raise Exception("Invalid model")


    return openai.ChatCompletion.create(
        engine=engine,
        messages=messages,
        temperature=0,
        max_tokens=1000,
        top_p=0,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )["choices"][0]["message"]["content"]


def claude(prompt, system, model="claude-1", api_key=None):
    # prompt=system + "\n\n" + prompt
    # prompt=prompt
    client = anthropic.Client(api_key)
    return client.completion(
        prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
        stop_sequences = [anthropic.HUMAN_PROMPT],
        model=model,
        max_tokens_to_sample=2000,
        temperature=0,
        top_p=0,
    )["completion"]



def baseline(source):
    result = ""
    for line in source.split("\n"):
        if line.strip():
            result += "> " + line + "\n"
        else:
            result += "\n"
    return result



# @ray.remote
def process(obj):
    try:
        source = obj["input"]
        if "gpt" in model:
            completion = gpt(source, system, model=model)
        elif model == "claude":
            completion = claude(source, system)
        elif model == "bard":
            completion = bard(source, system)
        
        
        # source = obj["method"]
        # completion = baseline(source)

        with open(outfile, "a") as f:
            obj["prediction"] = completion
            print(json.dumps(obj), file=f)
    except Exception as e:
        print("ERROR:", e)


# num_examples should be zero_shot, one_shot or multi_shot
num_examples = sys.argv[1]
model = sys.argv[2]
out_path = sys.argv[3]
infile = f'inputs/{num_examples}.jsonl'
outfile = out_path
# remove if the file exists
Path(outfile).unlink(missing_ok=True)


# read a jsonl file and return a list of dicts
with open(infile) as f:
    lines = []
    for line in f:
        obj = json.loads(line)
        lines.append(obj)
    # ray.get([process.remote(line) for line in lines])
    [process(line) for line in lines]

