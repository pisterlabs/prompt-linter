import openai
import requests
import json

"""
This class contains functions to connect to openAI and huggingface APIs and request an AI generated response.
"""

"""
prompt: prompt given to the model
length: length of the essay
returns response from charGPT to the given prompt
"""
def openAI(prompt, length):
    # API key needs to be set by creating an account
    openai.api_key = "API_KEY"
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=length,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=1
    )
    return response["choices"][0]["text"]

"""
prompt: prompt given to the model
length: length of the essay
returns response from hugging face model to the given prompt
"""
def huggingFace(prompt, length):
    # API token will be provided after setting up an account on hugging face
    API_TOKEN = ""
    # find a model on hugging face and set API_URL
    API_URL = ""
    # set length in headers
    headers = {"Authorization": f"Bearer {API_TOKEN}"}
    response = requests.post(API_URL, headers=headers, json=json.dumps(prompt))
    ret = json.dumps(response.json())
    # ret should be a python string and not in json format
    return ret