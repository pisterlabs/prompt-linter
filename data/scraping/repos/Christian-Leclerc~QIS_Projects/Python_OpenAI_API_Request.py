import openai
import os
import pandas as pd

API_KEY = os.environ["OpenAI_API"]
openai.api_key = API_KEY


def get_completion(prompt, model="gpt-3.5-turbo"):
    """
    Function to run api request to execute query (prompt) for AI using openai library.
    :param prompt: The query request for OpenAI.
    :param model: gpt-3.5-turbo --> Token limits: 40,000 TPM, Request and other limits: 3 RPM/200 RPD.
    :return: ChatGPT response.
    """
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        engine=model,
        # messages is an array of message objects: 'role' (either "system," "user," or "assistant"),
        # 'content' (the actual text of the message).
        messages=messages,
        # max_tokens limits the response to a certain number of tokens,
        # it can be used to control the length of the generated output.
        max_tokens=100,
        # Temperature is a parameter that controls the “creativity” or randomness of the text generated by GPT-3.
        # A higher temperature (e.g., 0.7) results in more diverse and creative output,
        # while a lower temperature (e.g., 0.2) makes the output more deterministic and focused.
        temperature=0)
    return response.choices[0].message["content"]


prompt = input("Enter your query: ")
response = get_completion(prompt)
print(response)

# EXAMPLE WITHOUT USING openai LIBRARY

# import requests
# headers = {
#     "Content-Type": "application/json",
#     "Authorization": f"Bearer {OPENAI_API_KEY}"
# }
# response_json = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json={
#     "model": "gpt-3.5-turbo",
#     "messages": [{"role": "user", "content": prompt}],
#     "temperature": 0
# }).json()
# print(response_json["choices"][0]["message"]["content"])
