import requests
from dotenv import load_dotenv
import openai
import os

def configure():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")

# Get your API key from https://beta.openai.com/account/api-keys

def generate_story(prompt, max_tokens=400):
    response = openai.Completion.create(
        engine="davinci",
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=0.8,
    )
    return response.choices[0].text

configure()
response = generate_story("Create a story of a lion and rabbit for a 4 year old.")
print("Your story is as follows ---> \n" +  response)


