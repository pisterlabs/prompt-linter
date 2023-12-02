"""
This module is used to test the Eco-Bot vision functionality.
"""
from IPython.display import display, Image, Audio

import cv2  # We're using OpenCV to read video
import base64
import time
import openai
import os
import requests
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load API keys from .env file
load_dotenv()

client = OpenAI()
client.api_key = os.getenv("OPENAI_API_KEY")

class EcoBot_Vision:
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
    # TODO - CHANGE the code to allow user to input the image, display the image and the response 

class EcoBot_Video_Vision:

    def run_video(self, video_path):
        video = cv2.VideoCapture(video_path)

        base64Frames = []
        while video.isOpened():
            success, frame = video.read()
            if not success:
                break
            _, buffer = cv2.imencode(".jpg", frame)
            base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

        video.release()
        print(len(base64Frames), "frames read.")

    def display_video(self, base64Frames):
        display_handle = display(None, display_id=True)
        for img in base64Frames:
            display_handle.update(Image(data=base64.b64decode(img.encode("utf-8"))))
            time.sleep(0.025)

    def prompt(self, base64Frames):
        PROMPT_MESSAGES = [
            {
                "role": "user",
                "content": [
                    "These are frames from a video that I want to upload. Generate a compelling description that I can upload along with the video.",
                    *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::10]),
                ],
            },
        ]
        params = {
            "model": "gpt-4-vision-preview",
            "messages": PROMPT_MESSAGES,
            "api_key": os.environ["OPENAI_API_KEY"],
            "headers": {"Openai-Version": "2020-11-07"},
            "max_tokens": 200,
        }

        result = openai.ChatCompletion.create(**params)
        print(result.choices[0].message.content)

    def create_narraitor(self, base64Frames):
        PROMPT_MESSAGES = [
            {
                "role": "user",
                "content": [
                    "These are frames of a video. Create a short voiceover script in the style of David Attenborough. Only include the narration.",
                    *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::10]),
                ],
            },
        ]
        params = {
            "model": "gpt-4-vision-preview",
            "messages": PROMPT_MESSAGES,
            "api_key": os.environ["OPENAI_API_KEY"],
            "headers": {"Openai-Version": "2020-11-07"},
            "max_tokens": 500,
        }

        result = openai.ChatCompletion.create(**params)
        print(result.choices[0].message.content)