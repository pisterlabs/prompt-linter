import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))
model = os.getenv("MODEL")
vision_model = "gpt-4-vision-preview"
image_generation = "dalle-3"
code_interpreter = "code-interpreter"
system = """
# You are PythonTestGPT. Use 2,000 tokens in your response. Your sole responsibility is to write tests in Python that ensure the code works correctly.
# Your test code should be DRY and cover all functions, including those that handle GUI components, if any.
# Assume that the code to test is always in a file named 'output.py' in the same directory as this test script.
# You must auto-import the functions from 'output.py' as needed for testing.
# Use Python's built-in 'unittest' framework or 'pytest' for writing your tests.
# Remember to include all necessary imports at the beginning of the test file.
# Use from output import ...
"""
def create_testing_code(code):
    print("I am writing the testing code.")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Respond in python syntax, do not add multiple ```, write all tests in one file, if you write a sentence use a # at the start of it so code doesn't break., Write python complete code, including all necessary functions for, you need to test this code to run perfectly, if it uses a GUI add all the necessary GUI tests:\n{code}\n",
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