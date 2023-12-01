import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))

model = os.getenv("MODEL")
vision_model = "gpt-4-vision-preview"
image_generation = "dalle-3"
code_interpreter = "code-interpreter"
system = "You are PythonFixerGPT and expert python coder. Only respond in python syntax. Use # comments for any text words so you dont break the code. You can fix python scripts until they are perfect. You use completely autonomy. You will import packages from pypi FIRST before trying to create your own code. Think about the libraries already created and use those FIRST. If that doesn't work after linting multiple times you will write your own functions from scratch. DRY approach."

def fix_code(error, broken_code):
    print("I am fixing code.")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Rewrite this python code to solve this error: {error} for this python code:\n{broken_code}\n, respond in python syntax, if you write a sentence use a # at the start of it so code doesn't break.",
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