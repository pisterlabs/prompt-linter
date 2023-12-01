import subprocess

import os
import traceback
import inspect
from functools import wraps
from dotenv import load_dotenv
from langchain.llms import OpenAI
import openai
# Load environment variables
load_dotenv()
# Initialize LangChain's LLM with GPT-4 using the API key from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
llm = OpenAI(api_key=OPENAI_API_KEY, model="gpt-4-1106-preview")
openai.api_key = OPENAI_API_KEY
md_response = []
def llm_debugger(reflections=1, output = 'error_response.md'):  # Set a default reflection count
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"Exception:'{e}' being sent to LLM for debugging")
                # Get the content of the function (source code)
                function_content = inspect.getsource(func)
                # Convert the traceback to a string
                error_message = traceback.format_exc()
                # Send to the LLM for debugging and reflection
                responses = send_to_llm_for_debugging(function_content, error_message, reflections)
                # Log the responses to a file
                #if output does not exist, create it
                
                with open(output, "w") as file:
                    file.write(f"Function `{func.__name__}` raised an exception:\n{error_message}\n")
                    for idx, response in enumerate(responses, 1):
                        file.write(f"LLM Reflection {idx}:\n{response}\n")
                    file.write("----------\n")
                raise  # Re-raise the exception after logging
        return wrapper
    return decorator

def send_to_llm_for_debugging(function_content, error_message, reflections):
    prompt = (f"Debug the following error:\n{error_message}\nin the function:\n{function_content}\n"
              "Include the full updated function in your response.")
    messages = [{"role": "system", "content": "You are a helpful assistant."}]
    md_responses = []  # Store each response for logging
    
    for i in range(reflections):
        messages.append({"role": "user", "content": prompt})
        response = openai.ChatCompletion.create(model="gpt-4", messages=messages)
        response_content = response.choices[0].message['content']
        md_responses.append(response_content)
        messages.append({"role": "assistant", "content": response_content})
        prompt = "Reflect on your previous answer and provide any corrections or additional insights if necessary."

    return md_responses

# Example usage with reflection parameter
#@llm_debugger(reflection=2)
#def test_function(x):
#    return 10 / x  # This will raise a ZeroDivisionError when x is 0
#
## Running the test_function with an argument that causes an error
#try:
#    test_function(0)
#except ZeroDivisionError:
#    pass  # Exception has already been caught, logged, and printed by decorator
