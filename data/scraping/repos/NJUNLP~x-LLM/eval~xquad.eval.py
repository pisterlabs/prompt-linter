# %%
import os
import re
import openai
import time
import json
from tqdm import tqdm
openai.api_key = "sk-1234567890qwertyuiop"
template = """You will be given a context followed by question. You will then be given one potential answer to the question.
Your task is to tell if the answer is correct.
Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Correctness (YES or NO): Is the answer correct?
YES means the answer provides an accurate and valid response that aligns with the facts, logic, and requirements of the question. The answer should be in the same language as the context.
NO means otherwise.

Context: {}
Answer: {}

Evaluation Form (YES or NO):
"""

# %%
def eval_file(file, end=int(1e9), start=0):
    print(f"[start evaluation]: L{start}-L{end} in {file}")
    lines = list(map(json.loads, open(file, encoding="utf-8").readlines()))
    if end < int(1e9) and len(lines) < end - start:
        print(f"[skip evaluation]: incomplete file {file}")
        return
    end = min(end, len(lines))
    cases = list(map(
        lambda x: (x["input"], x["prediction"]), 
        lines
    ))
    filename = f"{file}.{start}-{end}.eval"
    if os.path.exists(f"{filename}.log"):
        with open(f"{filename}.log", "r") as log:
            old = len(log.readlines())
            print(f"[resume evaluation]: L{start}-L{start + old}")
            start += old
    
    with open(f"{filename}.log", "a+", encoding="utf-8") as log, open(f"{filename}.txt", "a+", encoding="utf-8") as log_raw:
        for i, case in enumerate(tqdm(cases[start:end])):
            if len(case[1].strip()) == 0:
                log.write("NO\n")
            # time.sleep(0.5)
            try:
                completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": template.format(*case)}
                    ],
                    temperature = 0
                )
                content = completion.choices[0].message.content
            except openai.error.APIError:
                content = "NO(Error)"
            log_raw.write(template.format(*case) + "\n" + content + "\n")
            try:
                log.write(re.search(r"(YES)|(NO)", content).group(0) + "\n")
            except AttributeError:
                log.write("NO(Error)\n")
        
# %%
files = sys.argv[1:]
for file in tqdm(files):
    eval_file(file, end=100)
