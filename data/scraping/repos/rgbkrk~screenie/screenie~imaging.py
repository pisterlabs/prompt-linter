import base64
import io
import time

import cv2
import pyautogui
from openai import OpenAI
from PIL import Image

from .prompts import prompts

client = OpenAI()


def take_screenshot(scale_factor=0.2) -> str:
    screenshot = pyautogui.screenshot()

    new_size = (
        int(screenshot.size[0] * scale_factor),
        int(screenshot.size[1] * scale_factor),
    )

    print(new_size)

    image = screenshot.resize(new_size, Image.LANCZOS)

    # Convert to WebP or PNG format in memory
    buffer = io.BytesIO()
    image.save(buffer, format="WEBP")
    buffer.seek(0)

    # Encode the image in base64
    base64_image = base64.b64encode(buffer.read()).decode("utf-8")
    return base64_image


def take_picture(scale_factor=0.2) -> str:
    # Initialize the webcam
    cap = cv2.VideoCapture(0)

    try:
        # Check if the webcam is opened correctly
        if not cap.isOpened():
            raise IOError("Cannot open webcam")

        # Capture and discard a few frames to let the camera warm up
        for _ in range(10):
            ret, frame = cap.read()
            time.sleep(0.001)

        ret, frame = cap.read()
        if not ret:
            raise IOError("Cannot capture image from webcam")

        new_size = (
            int(frame.shape[1] * scale_factor),
            int(frame.shape[0] * scale_factor),
        )

        frame = cv2.resize(
            frame, dsize=new_size, interpolation=cv2.INTER_AREA
        )

        # Convert the image to a byte array
        is_success, buffer = cv2.imencode(".jpg", frame)
        if not is_success:
            raise IOError("Cannot convert image to byte array")

        # Encode the byte array to base64
        base64_str = base64.b64encode(buffer.tobytes()).decode()
        return base64_str

    finally:
        # Release the webcam in the 'finally' block to ensure it's always executed
        cap.release()


def create_image_description_message(base64_image: str):
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


def analyze_image(base64_image: str, script, prompt="attenborough"):
    # If the named prompt is chosen, use it. Otherwise assume it's the actual prompt
    system_prompt = prompts.get(prompt, prompt)

    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
        ]
        + script
        + create_image_description_message(base64_image),
        max_tokens=500,
    )
    response_text = response.choices[0].message.content
    return response_text
