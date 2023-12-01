import openai
import os

"""
load environment variables from .env file
"""
from dotenv import load_dotenv
load_dotenv()

"""
access the openai api environment variable
"""
openai.api_key = os.getenv("OPENAI_API_KEY")

"""
image request controller
"""
def get_image(text):
    try:
        response = openai.Image.create(
            prompt = f"{text}",
            n=1, # batch size per response
            size="1024x1024"
        )

    except openai.error.OpenAIError as e:
        print(e.http_status)
        print(e.error)

    return response['data'][0]['url']
