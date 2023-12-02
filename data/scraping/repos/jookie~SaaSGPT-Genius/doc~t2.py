import time
# import openai
from openai import OpenAI
import os

api_key = "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"
client = OpenAI(api_key=api_key)
# Set the API key for the OpenAI client
client.api_key = api_key

# export OPENAI_API_KEY="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"
# Set the API key as an environment variable
os.environ["OPENAI_API_KEY"] = api_key

# Set your API key
client = OpenAI()

# Set the API key for the OpenAI client
client.api_key = api_key




client = OpenAI()

response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What’s in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    },
                },
            ],
        }
    ],
    max_tokens=300,
)

print(response.choices[0])