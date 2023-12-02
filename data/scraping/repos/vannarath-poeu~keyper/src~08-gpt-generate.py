import openai
import dotenv
import os
import argparse
import pathlib
import json
import re
import time

def generate():
    test_jsonl = f"data/midas/ldkp3k/test.jsonl"
    data_path = "output/gpt/raw"
    output_path = "output/gpt/processed"

    assert os.path.exists(test_jsonl), f"File {test_jsonl} does not exist"

    if not os.path.exists(f"{output_path}"):
        pathlib.Path(f"{output_path}").mkdir(parents=True, exist_ok=True)
    
    # with open(test_jsonl, "r") as f:
    #     test = f.readlines()
    
    # num_records = len(test)

    files = os.listdir(data_path)
    result = {}
    for file in files:
        i = file.split(".")[0]
        phrases = []
        with open(f"{data_path}/{file}", "r") as f:
            record = f.readlines()
            start = False
            for line in record:
                if line.startswith("Task 2:"):
                    start = True
                elif len(phrases) >= 10 or line.startswith("Task 3:"):
                    break
                elif start:
                    phrase = re.sub("-", "", line)
                    phrase = re.sub("\d+\.", "", phrase)
                    phrase = phrase.strip().lower()
                    if phrase:
                        phrases.append(phrase)
        result[i] = phrases
    
    with open(f"{output_path}/keyphrases.json", "w") as f:
        json.dump(result, f)


if __name__ == "__main__":
    # Example: python3 src/08-gpt-generate.py

    generate()