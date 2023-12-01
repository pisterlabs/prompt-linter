import json
import openai
import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored

GPT_MODEL = "gpt-3.5-turbo-0613"

"""
How to call functions with chat models

Ref:
    https://github.com/openai/openai-cookbook/blob/main/examples/How_to_call_functions_with_chat_models.ipynb
"""


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, functions=None, function_call=None, model=GPT_MODEL):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai.api_key,
    }
    json_data = {"model": model, "messages": messages}
    if functions is not None:
        json_data.update({"functions": functions})
    if function_call is not None:
        json_data.update({"function_call": function_call})
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


# ex - 1
messages = []
messages.append({
    "role": "system",
    "content": "Don't make assumptions about what values to plug into functions. "
               "Ask for clarification if a user request is ambiguous."
})
messages.append({"role": "user", "content": "What's the weather like today"})
chat_response = chat_completion_request(
    messages
)
assistant_message = chat_response.json()["choices"][0]["message"]
print(assistant_message)
"""
Output of `assistant_message`:
{'role': 'assistant', 'content': "I'm sorry, but you didn't mention the location. Can you please provide me with the name of the city or town?"}
"""
messages.append(assistant_message)

# ex -2 (add the function)
# TODO:

print()
