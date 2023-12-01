import cv2
import base64
import time
from openai import OpenAI
import os
from pathlib import Path

client = OpenAI()

def say(text: str):
    speech_file_path = Path(__file__).parent / "data" / "narrate.mp3"

    response = client.audio.speech.create(
        model="tts-1",
        voice="fable",
        input=text
    )

    response.stream_to_file(speech_file_path)

video = cv2.VideoCapture("data/big_buck_bunny_720p.mp4")

base64Frames = []
while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

video.release()

print(len(base64Frames), "frames read.")

PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            "These are frames from a video that I want to upload. Create a short voiceover script in the style of David Attenborough. Only include the narration.",
            *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::480]),
        ],
    },
]

result = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=PROMPT_MESSAGES,
    max_tokens=200,
)

narration = result.choices[0].message.content

print(narration)

say(narration)