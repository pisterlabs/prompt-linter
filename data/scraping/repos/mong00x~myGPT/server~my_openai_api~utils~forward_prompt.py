# use openai library

import openai
import os
from dotenv import load_dotenv

load_dotenv( dotenv_path = ".env.local")

openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_response(prompt):
    print(openai.api_key)
    
    response = openai.Completion.create(
        engine="text-davinci-002", # replace with your engine
        prompt=prompt,
        temperature=0.5,
        max_tokens=100
    )
    return response.choices[0].text.strip()
