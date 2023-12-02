import cv2
import base64
import time
import openai
import requests
import os
import argparse
from dotenv import dotenv_values, load_dotenv
import time

config = dotenv_values("/workspace/Research/PangyoPangyo/src/.env")

openai.organization = config.get('OPENAI_ORGANIZATION')
openai.api_key = config.get('OPENAI_API_KEY')

### Define the argument parser

def define_argparser():
    p = argparse.ArgumentParser()

    p.add_argument("--data_path", type=str, required=True)
    
    config = p.parse_args()

    return config


def main(config):
    # Ensure the dataset directory exists and has the video file
    if not os.path.exists(config.data_path):
        print("Video file not found. Make sure data_path exists.")
        return

    video = cv2.VideoCapture(config.data_path)

    base64Frames = []
    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

    video.release()
    print(len(base64Frames), "frames read.")

    # Skipping the display part as it's not relevant in a .py script

    INSTRUCTOIN = " ".join(
        "These are frames of a video.",
        "Create a short voiceover script in the style of a super excited brazilian sports narrator who is narrating his favorite match.",
        "He is a big fan of Messi, the player who scores in this clip.",
        "Use caps and exclamation marks where needed to communicate excitement.",
        "Only include the narration, your output must be in english.",
        "When the ball goes into the net, you must scream GOL either once or multiple times."
    )

    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                INSTRUCTOIN,
                *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::10]),
            ],
        },
    ]
    params = {
        "model": "gpt-4-vision-preview",
        "messages": PROMPT_MESSAGES,
        "api_key": openai.api_key,
        "headers": {"Openai-Version": "2020-11-07"},
        "max_tokens": 500,
    }

    result = openai.ChatCompletion.create(**params)
    print(result.choices[0].message.content)


if __name__ == "__main__":
    config = define_argparser()
    main(config)
