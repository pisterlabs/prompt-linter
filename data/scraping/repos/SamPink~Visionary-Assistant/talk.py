import os
import numpy as np
import pyautogui
import cv2
import base64
import requests
from openai import OpenAI
from playsound import playsound
from dotenv import load_dotenv


def capture_and_process_screen():
    """Capture and process the screen into a JPEG format."""
    success, buffer = cv2.imencode(".jpg", np.array(pyautogui.screenshot()))
    if not success:
        raise RuntimeError("Failed to encode screenshot.")
    return buffer


def analyze_image(buffer, api_key):
    """Analyze the image using OpenAI's API and return the analysis."""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What do you see in this image? Explain in a couple sentences.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}"
                        },
                    },
                ],
            }
        ],
        "max_tokens": 300,
    }
    response = requests.post(
        "https://api.openai.com/v1/chat/completions", headers=headers, json=payload
    ).json()
    if "error" in response:
        raise ValueError(response["error"]["message"])
    return response["choices"][0]["message"]["content"]


def main():
    load_dotenv()
    client = OpenAI()  # Ensure your API key is correctly configured
    api_key = os.getenv("OPENAI_API_KEY")

    while True:
        try:
            buffer = capture_and_process_screen()
            analysis = analyze_image(buffer, api_key)
            print("🎙️ Analysis:", analysis)

            speech_path = "speech.mp3"
            client.audio.speech.create(
                model="tts-1", voice="alloy", input=analysis
            ).stream_to_file(speech_path)
            playsound(speech_path)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
