import re
from dotenv import load_dotenv
import os
import json
from chromadb.config import Settings
from chromadb.utils import embedding_functions
import chromadb
import openai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)
input_path = "C:\@code\APIMISUSE\data\detection_result_yes_v3.json"
output_path = "C:\@code\APIMISUSE\data\generate_fix_yes_v3.json"


data = []
with open(input_path, encoding="utf-8") as f:
    data = json.load(f)
print(len(data))
print(data[0].keys())

template_2 = """

Please read the following code snippet and fix rules. Then, think step by step and answer if the fix pattern can be applied in the code snippet.
If pattern can be applied, generate the fixed code snippet. If not, please answer "No" in Decision and answer NA in Fixed.

Code snippet:
{code_before}

Fix rules:
{fix_rules}

Think steps: (please be concise)
Decision: (Yes/No)
Fixed: (generate fixed code)
"""
# 14
for i in range(20, len(data)):
    print("current_index:", i, "/", len(data))
    number = data[i]["number"]
    code_before = data[i]["code_before"]
    fix_pattern_prompt = data[i]["example"]
    prompt_2 = template_2.format(
        code_before=code_before, fix_rules=fix_pattern_prompt)
    print("prompt_2", len(prompt_2))

    # decition = openai.ChatCompletion.create(
    Fixed = completion_with_backoff(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt_2}
        ]
    )
    Fixed = Fixed["choices"][0]["message"]["content"]

    output = {
        "number": number,
        "code_before": code_before,
        "example": fix_pattern_prompt,
        "Fixed": Fixed,
        "prompt_2": prompt_2,
    }

    with open(output_path, 'a') as f:
        json.dump(output, f)
        f.write(os.linesep)
