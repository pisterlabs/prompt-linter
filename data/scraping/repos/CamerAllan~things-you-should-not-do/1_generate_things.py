import os
from typing import Dict, List
import openai
import json
import argparse
import random
from helpers.helpers import load_samples, load_unused, random_line

openai.api_key = os.getenv("OPENAI_API_KEY")

parser = argparse.ArgumentParser()
parser.add_argument("data_folder", help="path to data folder")
parser.add_argument("-n", type=int, default=10)

# Parse and print the results
args = parser.parse_args()
data_folder = args.data_folder
if data_folder is None:
    print("Uh oh!")
    exit(1)

adjectives_filepath = f"{data_folder}/prompt-adjectives.txt"
n = args.n

# BEGIN LOGIC -----------------------------------------------------------------

unused = load_unused(data_folder)
samples = load_samples(data_folder)


def generate_prompt_examples_sample(n: int):
    return random.sample(samples, n)


def generate_prompt(prompt_examples_sample):
    with open(adjectives_filepath, "r") as adjectives_file:
        adjective_1 = random_line(adjectives_file).strip()

    prompt_start = {
        "role": "system",
        "content": (
            'You generate suggestions for XKCD\'s "Things You Should Not Do" comic.\n'
            "Suggestions should be in the style of Randall Munroe's writing.\n"
            "Suggestions should always be humerous.\n"
            "Suggestions should always be in the imperative tense.\n"
            "Good suggestions might be outrageous, physically improbable or dangerous."
        ),
    }

    prompt = [prompt_start]
    for sample_example in prompt_examples_sample:
        prompt.append({"role": "assistant", "content": sample_example})

    return prompt


def generate(prompt: List[Dict], n) -> str:
    ideas = []

    response = openai.ChatCompletion.create(
        model="gpt-4", messages=prompt, temperature=1.1, n=n, presence_penalty=1
    )

    ideas = [
        r["message"]["content"].replace(".", "").strip()
        for r in response.get("choices")
    ]

    print("\n".join(ideas))
    return ideas


def generate_n(n: int):
    samples = generate_prompt_examples_sample(n)
    prompt = generate_prompt(samples)
    things = generate(prompt, n)

    return things


unused = generate_n(n)

with open(f"{data_folder}/unused.json", "w") as file:
    file_things = json.dump(unused, file, indent=4)
