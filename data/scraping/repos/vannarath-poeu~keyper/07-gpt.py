import openai
import dotenv
import os
import argparse
import pathlib
import json
import re
import time
from openai.error import InvalidRequestError, OpenAIError

def generate(records, i, output_path, num_keyphrases):
    record = json.loads(records[i])

    context = ""
    for j, section in enumerate(record["sections"]):
        if section.lower() == "abstract":
            continue

        tokens = re.sub(r'[^a-zA-Z\s]', '', " ".join(record["sec_text"][j]))
        tokens = re.sub(r'\s\s+', ' ', tokens)
        tokens = tokens.strip()

        context += "Section name: " + section + "\n"
        context += "Section text: " + tokens + "\n\n"

    instruction = f"""
Do the followings:
1. For each section in the above paper, generate up to {num_keyphrases} key noun phrases without prepositions and conjunctions.
2. Generate up to {num_keyphrases} key noun phrases, without prepositions and conjunctions and order by relevance for the whole paper.
3. Generate an abstract for this paper.

For task 1 and 2, try to avoid using prepositions and conjunctions.
"""

    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
            {"role": "system", "content": "You're the author of the following scientific paper."},
            {"role": "user", "content": context},
            {"role": "user", "content": instruction}
        ],
        temperature=0,
    )
    response = chat_completion.choices[0].message.content

    with open(f"{output_path}/{i}.txt", "w") as f:
        f.write(response)

def gpt(
    output_path: str,
    num_keyphrases: int,
):
    test_jsonl = f"data/midas/ldkp3k/test.jsonl"

    assert os.path.exists(test_jsonl), f"File {test_jsonl} does not exist"

    if not os.path.exists(f"{output_path}"):
        pathlib.Path(f"{output_path}").mkdir(parents=True, exist_ok=True)
    
    with open(test_jsonl, "r") as f:
        test = f.readlines()
    
    num_records = len(test)
    print(f"Number of records: {num_records}")

    for i in range(num_records):
        generate(test, i, output_path, num_keyphrases)
        # Prevent throttling
        time.sleep(1)

if __name__ == "__main__":
    # Example: python3 src/06-maxflow-activation.py
    parser = argparse.ArgumentParser()
    # Add list of arguments
    parser.add_argument("--data", type=str, default="output/midas/ldkp3k")
    parser.add_argument("--output", type=str, default="output/gpt/raw")

    args = parser.parse_args()
    # Get all the variables
    data_path = args.data
    output_path = args.output

    # Read the OPENAI_API_KEY from .env file
    load_env = dotenv.load_dotenv()

    if not load_env:
        raise Exception("Could not load .env file")

    openai.api_key = os.environ["OPENAI_API_KEY"]

    num_keyphrases = 20

    gpt(
        output_path,
        num_keyphrases=num_keyphrases,
    )