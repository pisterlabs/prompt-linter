import openai
import os
import dotenv
import json
import sys
import inspect

# read these for more info:
# https://chat.openai.com/share/9341df1e-65c0-4fd8-87dc-80e0dc9fa5bc
# https://chat.openai.com/share/193482c6-e7b3-4022-b42b-cd2530efb839

dotenv.load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

async def run_chat_prompt(prompt, model="gpt-4"):
    completion = openai.ChatCompletion.create(
    model=model,
    messages=[
        {"role": "user", "content": prompt}
    ]
    )

    return (completion.choices[0].message.content)

def wrap_error(e: Exception, description: str = None):

    exc_type, exc_value, exc_traceback = sys.exc_info()

    filename = exc_traceback.tb_frame.f_code.co_filename

    line_number = exc_traceback.tb_lineno

    func_name = exc_traceback.tb_frame.f_code.co_name

    module = inspect.getmodule(exc_traceback.tb_frame)
    function_obj = getattr(module, func_name)

    # Get and print the source code of the function
    func_code = inspect.getsource(function_obj)

    error_message = f"This error message occurred because of '{str(e)}' at line {line_number} in file {filename}, function {func_name}. The exception type is {exc_type}. The function is: \n\n```python\n{func_code}\n```\n\nHow do I fix this?"
    if description:
        error_message += f'Other details: {description}'

    print(error_message + '\n\n')

    return error_message