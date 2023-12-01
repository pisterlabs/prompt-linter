import os
from openai import OpenAI
import base64
import json
import time
import simpleaudio as sa
import errno
from elevenlabs import generate, play, set_api_key, voices
import subprocess

api_key = "<your key>"
client = OpenAI(api_key=api_key)

# your openai key
set_api_key("<your key>")

def encode_image(image_path):
    while True:
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        except IOError as e:
            if e.errno != errno.EACCES:
                # Not a "file in use" error, re-raise
                raise
            # File is being written to, wait a bit and retry
            time.sleep(0.1)


# your elevenlabs voice key
def play_audio(text):
    audio = generate(text, voice="<your key>")

    unique_id = base64.urlsafe_b64encode(os.urandom(30)).decode("utf-8").rstrip("=")
    dir_path = os.path.join("narration", unique_id)
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, "audio.wav")

    with open(file_path, "wb") as f:
        f.write(audio)

    open_with_quicktime(file_path)


# open with a quicktime player
def open_with_quicktime(file_path):
    quicktime_path = "/System/Applications/QuickTime Player.app"
    subprocess.run(["open", "-a", quicktime_path, "-j", file_path])

# Example usage:
# Replace 'your_audio_file.wav' with the actual path to your generated audio file
latest_file_path = None

# Play the latest recording
if latest_file_path is not None:
    open_with_quicktime(latest_file_path)


def generate_new_line(base64_image):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image"},
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                },
            ],
        },
    ]


def analyze_image(base64_image, script):
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": """
                You are Morgan Freeman. Narrate human, as if you are describing a good friend with a bright future ahead. Make it punchy, and make it sound like you care about this person. Don't repeat yourself. If I do anything remotely unexpected, describe how this ties into my greater aspiring character.
                """,
            },
        ]
        + script
        + generate_new_line(base64_image),
        max_tokens=500,
    )
    response_text = response.choices[0].message.content
    return response_text


def main():
    script = []

    while True:
        # path to your image
        image_path = os.path.join(os.getcwd(), "./frames/frame.jpg")

        # getting the base64 encoding
        base64_image = encode_image(image_path)

        # analyze posture
        print("👀 Morgan Freeman is watching...")
        analysis = analyze_image(base64_image, script=script)

        print("🎙️ Morgan Freeman says:")
        print(analysis)

        play_audio(analysis)

        script = script + [{"role": "assistant", "content": analysis}]

        # wait for 5 seconds
        time.sleep(20)


if __name__ == "__main__":
    main()
