"""
alphas.py

Given an set of collected language utterances for various tasks (all in `data/language/*.json`) run the GPT-3 alpha
prediction, using the prompt and in-context examples specified below.

Note: We use the (paid) GPT-3 API, using `text-davinci-002` to produce these alpha values. This requires an API key,
that you need to export as an environment variable $OPENAI_API_KEY to use this script! Any charges should be minimal.
"""
import json
import os
from pathlib import Path

import openai
from tqdm import tqdm


# Set OpenAI API Key (from Environment Variable)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Path to *all* language utterances
LANGUAGE_UTTERANCES = Path("data/language")

# === Important: this is the GPT-3 Prompt we used for the *entirety* of the LILAC work ===
PROMPT = (
    "I'm building a robot that can follow language commands. Tell me (YES or NO) if the robot can execute "
    "the following language instructions without knowing any other information about its environment.\n\n"
    "Input: move to the right\n"
    "Output: YES\n\n"
    "Input: rapidly twist to the front\n"
    "Output: YES\n\n"
    "Input: clean up the spilled coffee\n"
    "Output: NO\n\n"
    "Input: left\n"
    "Output: YES\n\n"
    "Input: move towards the bookshelf\n"
    "Output: NO\n"
    "Input: %s\n"
    "Output:"
)


def gpt3_alphas() -> None:
    # Assemble list of all utterances...
    utterances, utterance2alpha = [], {}
    for task_file in LANGUAGE_UTTERANCES.iterdir():
        with open(task_file, "r") as f:
            utterances.extend(json.load(f)["utterances"])

    # Iterate through utterances, and feed to GPT-3, filling in the above prompt with the given example to tag...
    print("[*] Annotating all Language Utterances w/ Alphas using GPT-3...")
    for u in tqdm(utterances):
        response_json = openai.Completion.create(model="text-davinci-002", prompt=PROMPT % u, max_tokens=4, stop="\n")

        # Parse out the text response --> switch on YES (alpha = 0) / NO (alpha = 1)
        gpt3_output = response_json["choices"][0]["text"].strip()
        if gpt3_output.lower() in {"yes"}:
            utterance2alpha[u] = 0.0
        elif gpt3_output.lower() in {"no"}:
            utterance2alpha[u] = 1.0
        else:
            raise ValueError("Should return `yes` or `no`!")

    # Serialize...
    with open("data/gpt3-alphas.json", "w") as f:
        json.dump(utterance2alpha, f, indent=4)


if __name__ == "__main__":
    gpt3_alphas()
