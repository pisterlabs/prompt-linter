import os
from pylint import epylint as lint
from io import StringIO
import importlib
import re
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))

model = os.getenv("MODEL")
vision_model = "gpt-4-vision-preview"
image_generation = "dalle-3"
code_interpreter = "code-interpreter"
system = "You are PythonImproverGPT and expert python coder. Only respond in python syntax. Use # comments for any text words so you dont break the code. You can develop python scripts until they are perfect. You use completely autonomy. You will import packages from pypi FIRST before trying to create your own code. Think about the libraries already created and use those FIRST. If that doesn't work after linting multiple times you will write your own functions from scratch. DRY Approach."


def improve_code(code):
    print("I am improving the code.")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Respond in python syntax, if you write a sentence use a # at the start of it so code doesn't break., Write python complete code, including all necessary functions for, you need to improve this code to be as DRY as possible:\n{code}\n",
            },
        ],
        temperature=0.7,
        max_tokens=2500,
        stop=None,
        n=1,
        presence_penalty=0,
        frequency_penalty=0,
    )
    code = response.choices[0].message.content
    # Get the index of the first occurrence of "```python"
    try:
        start_index = code.index("```python") + len("```python") + 1

        # Get the index of the last occurrence of "```"
        end_index = code.rindex("```")

        # Extract the Python code between the start and end indices
        code = code[start_index:end_index]
        return code
    except ValueError:
        print("I had an problem parsing the code so it might contain errors.")
        return code

