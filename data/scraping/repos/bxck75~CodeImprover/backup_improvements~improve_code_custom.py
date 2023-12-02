
# TODO: add a wrapper method that can iterate x steps on the same script
# TODO: add a method that can improve all files in a given directory

import sys, re
from pathlib import Path
from os import path
import g4f

#protected
sys.path.append(str(Path(__file__).parent.parent.parent))

#protected
def read_code(text):
    match = re.search(r"```(python|py|)\n(?P<code>[\S\s]+?)\n```", text)
    if match:
        return match.group("code")
    
path = input("Path: ")

with open(path, "r") as file:
    code = file.read()

ingesting_and_analizing_role = f"""You Are a pragmatic, procedural and organized CodeAnalizing and CodeImproving Agent.
                                    Its the year 3145 and your Task is to ingest,analyze and improvea scripts. """

alter_prompt = f"""
{ingesting_and_analizing_role}
Improve the code in this file:
```py
{code}
```
Task:
    Add methods, enums or classes if needed.
    Add typehints if possible.
    Add a llm-agent (G4FLLM) to automate tasks by adding 'from langchain_g4f import G4FLLM' if possible.
    Refactor the code into a fully OOP callable class with __name__ == :__main__:
    Look for lines starting with # TODO: and try to add the functionality described there as a seperate experimental try-except block under the  __name__ == :__main__:
    Add a Usage comment block at the end of the script explaining the usage of all fuctionalities of the script 
    Add predicted use cases in a comment block
    Propose extra features in a comment block
    Add a pretty docstring at the beginning of the script

Rules:
    Don't change methods or classes that have '#protected' above the definition
    Don't remove anything only add functionality
    Don't remove imports of the script 
    All new and experimental code in a try-except block
    Don't add any typehints to kwargs.
    Don't remove license comments.
    """

print(f"Create code for script at path:{path}...")

response = []
for chunk in g4f.ChatCompletion.create(
    model=g4f.models.gpt_35_long,
    messages=[{"role": "user", "content": alter_prompt}],
    timeout=3000,
    stream=False
):
    response.append(chunk)
    print(chunk, end="", flush=True)

response = "".join(response)

code = read_code(response)
if code:
    with open(path.replace(".","_generated_improvement."), "w") as file:
        file.write(code)

