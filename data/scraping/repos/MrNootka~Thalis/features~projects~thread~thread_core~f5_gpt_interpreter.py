import json
import sys
import os
import json
import os
import openai
from dotenv import load_dotenv
os.chdir(os.path.dirname(os.path.abspath(__file__)))
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def handbook_system():
    with open("handbook.json", "r") as file:
        handbook_data = json.load(file)
    
    system_data = handbook_data["system"]
    print("\nhandbook_system:",system_data)
    return system_data


def gpt4_api(x):
    # Make the OpenAI API call to get the response using GPT-4
    # handle if situations to change the prompt to send to gpt4 besed on which value is "status" from memory.json
    h_system = handbook_system()
    completion = openai.ChatCompletion.create(
    model="gpt-4",
    messages=[
        {"role": "system", "content": h_system},
        {"role": "user", "content": x}
    ]
    )


    return completion.choices[0].message





