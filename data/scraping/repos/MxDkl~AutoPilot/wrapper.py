import os
import base64
import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OpenAI.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI()


class wrapper:
    def __init__(self):
        self.api_key = OpenAI.api_key
        self.chat_history = []

    def vision(self, img_path, prompt):
        # base64 encode image
        with open(img_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

        # send request to openai
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{encoded_string}",
                        },
                    ],
                }
            ],
            max_tokens=500,
        )

        # update chat history


        return response.choices[0].message.content