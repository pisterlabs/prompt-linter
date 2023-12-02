
import base64
import time
import openai
import os
import requests
import config 
import io
import cv2


image_path = "image.jpg"  # Replace with the path to your image
frame = cv2.imread(image_path)

# Encode the image as a base64 string
_, buffer = cv2.imencode(".jpg", frame)
base64Image = base64.b64encode(buffer).decode("utf-8")


PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            '''
            These are frames of a video. Create a short voiceover script in the style of David Attenborough. Only include the narration.
            '''
            ,
            {"image": base64Image, "resize": 768,"detail": "high"},
        ],
    },
]
params = {
    "model": "gpt-4-vision-preview",
    "messages": PROMPT_MESSAGES,
    "api_key": config.openai_api_key,
    "headers": {"Openai-Version": "2020-11-07"},
    "max_tokens": 500,
}

result = openai.ChatCompletion.create(**params)
print(result.choices[0].message.content)

