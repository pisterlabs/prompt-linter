from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

import argparse
import glob
import itertools
import json
import os.path
import sys
import tqdm

from llm_test_helpers import get_llm, get_args
args = get_args(sys.argv)
llm = get_llm(args.model)
prompt_1 = """Function:
```py
def add(a, b):
    return a + b
```
Signature: 
```
{
"function_prototype": {
    "function_name": "add",
    "parameters": [{"name": "a", "type": "int"}, {"name": "b", "type": "int"}],
    "return_values": [{"type": "int"}]
}
}
```
Function:
```py"""

prompt_2 = """```
Signature: """

for function in tqdm.tqdm(itertools.chain(glob.glob("functions/*.py"))):
    if "__init__" in function:
        continue

    filename = function.replace("functions/", "").replace(".py", "")
    if os.path.isfile(f"problems/problem_{filename}.json"):
        print(f"Skipping {filename}, definition already exists.")
        continue

    # Get code and create prompt
    with open(function) as f:
        code = f.read()
    
    # Generate tests and clean up output
    output = llm.predict(prompt_1 + code + prompt_2)
    output = output.replace("```json", "").replace("```py", "").replace("```", "").strip()
    
    print(output)
    function_proto = json.loads(output)
    data = {
        "identifier": filename,
        "description": filename,
        "prompts": [
            {
                "prompt_id": "prompt",
                "prompt": "Generate tests in Python (compatible with pytest) that produce 100% code coverage. Output only Python code and nothing else before or after.",
                "input_code": code
            }
        ],
        "tags": ["Code Coverage"]
    }

    data = {**data, **function_proto}
    # Write out final code
    with open(f"problems/problem_{filename}.json", "w") as f:
        f.write(json.dumps(data, indent=4))