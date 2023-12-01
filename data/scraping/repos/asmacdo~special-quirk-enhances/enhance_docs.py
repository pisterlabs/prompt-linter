#!/usr/bin/python3
import os
import openai
import json
import sys

TARGET_KEY = "target file"
PROMPT_KEY = "prompt"
ENGINE = "text-davinci-002"


def find_info_after_string(search, body):
    lines = body.split('\n')
    for line in lines:
        if search in line:
            info = line.split(':', 1)[-1].strip()  # Extract the information after the colon
            return info
    raise Exception("Required string `{search}:` not found in text.")

def read_and_rewrite(target_file, user_prompt):
    with open(target_file, "r") as target:
        base_document = target.read()

    prompt = f"Given <PROMPT>{user_prompt}</PROMPT>, return an edited version of the following text. <DOC>{base_document}</DOC>"

    # TODO Raises: openai.error.AuthenticationError
    response = openai.Completion.create(
        engine=ENGINE,
        prompt=prompt,
        max_tokens=500
    )
    # Clobber quirk with enhanced
    with open(target_file, "w") as target:
        # target.write(str(response))
        target.write(response.choices[0].text.strip())

if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise Exception("Invocation: ./enhance_docs.py <issue_body>")
    issue_body = sys.argv[1]
    target_file = find_info_after_string(TARGET_KEY, body)
    user_prompt = find_info_after_string(PROMPT_KEY, body)

    openai.api_key = os.getenv('OPENAI_API_KEY')
    if not openai.api_key:
        raise Exception("OPENAI_API_KEY must be set.")

    read_and_rewrite(target_file, user_prompt)
