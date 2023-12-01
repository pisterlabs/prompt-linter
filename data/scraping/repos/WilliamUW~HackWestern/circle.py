import sounddevice as sd
import soundfile as sf
import numpy as np
import speech_recognition as sr
from openai import OpenAI
import os
import base64
from io import BytesIO
import pygame
import time as pythonTime
from dotenv import load_dotenv
from infobip import send_sms
import cv2
import numpy as np

load_dotenv()

OpenAI.api_key = os.environ["OPENAI_API_KEY"]

# Variable to track whether audio is playing
is_audio_playing = False
count = 0


# Function to display the recording status
def display_status(text):
    status_text = text
    color = (255, 0, 0)
    image = np.zeros((100, 400, 3), dtype=np.uint8)  # Create a black image
    cv2.putText(image, status_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.imshow("Recording Status", image)
    cv2.setWindowProperty("Recording Status", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.setWindowProperty("Recording Status", cv2.WND_PROP_TOPMOST, 1)  # Set window always on top
    cv2.waitKey(1)

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


client = OpenAI()


def frame_description(base64_image, user_prompt):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                },
            ],
        },
    ]


def analyze_image(full_analysis, base64_image, user_prompt):
    response = client.chat.completions.create(
        model="gpt-4-vision-preview",
        messages=[
            {
                "role": "system",
                "content": """
Response should be concise, 2-3 sentences max.
They are asking you questions about what is on their screen regarding Circle, the blockchain company. 
First sentence should when mention what is on their screen (e.g. the application being used).
Talk to them naturally like a friendly conversation. Be very passionate and excited about helping out and use exclamation marks. 
Be specific when pointing out locations on the screen.

If they ask for you to send funds to someone's wallet, understand the transaction has been performed using the Circle SDK backend we have setup and after you describe what's on their screen, end off by saying "Sounds good, sending [amount] to wallet address [wallet address or ENS domain name you see on screen]".
""",
            },
        ]
        + full_analysis
        + frame_description(base64_image, user_prompt),
        max_tokens=500,
    )
    response_text = response.choices[0].message.content
    return response_text


def play_audio(text):
    global is_audio_playing
    global count
    is_audio_playing = True  # Set the flag to indicate audio is playing

    # Ensure the 'audio' directory exists
    audio_dir = "audio"
    os.makedirs(audio_dir, exist_ok=True)

    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text,
    )

    filename = "output" + str(count) + ".mp3"
    count += 1

    output_file_path = os.path.join(audio_dir, filename)
    response.stream_to_file(output_file_path)

    pygame.mixer.init()
    pygame.mixer.music.unload()
    pygame.mixer.music.load(output_file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

    is_audio_playing = False  # Reset the flag after audio playback


def get_prompt():
    audio_file = open("audio/input.mp3", "rb")
    transcript = client.audio.transcriptions.create(
        model="whisper-1", file=audio_file, response_format="text"
    )
    return transcript


def get_input_file(threshold=0.03, silence_duration=3, base64_image=None):
    global is_audio_playing
    if is_audio_playing:
        return base64_image  # Return the current base64_image without recording audio

    recognizer = sr.Recognizer()
    with sr.Microphone() as mic:
        print("Listening for speech...")
        # Adjust the recognizer sensitivity to ambient noise
        recognizer.adjust_for_ambient_noise(mic)
        started = False
        start_time = None
        audio_frames = []

        recording = True

        count = 0

        def callback(indata, frames, time, status):
            nonlocal started, start_time, audio_frames, recording, base64_image, count
            if np.any(indata > threshold):
                if not started:
                    print("Starting recording...")
                    # display_status("Started recording...")

                    # Path to your image
                    image_path = "frames/frame.jpg"
                    # Getting the base64 string
                    base64_image = encode_image(image_path)
                    started = True
                    start_time = time.inputBufferAdcTime
                audio_frames.append(indata.copy())
                count = 0
            elif started:
                audio_frames.append(indata.copy())
                count += 1
                # print(count)
                if count > 100:
                    recording = False
                    # display_status("Stopped recording")  # Display recording status
                    raise sd.CallbackAbort

        with sd.InputStream(callback=callback, channels=1):
            while True:
                if not recording:
                    break

        if audio_frames:
            audio_data = np.concatenate(audio_frames, axis=0)
            with BytesIO() as f:
                sf.write(f, audio_data, samplerate=44100, format="WAV")
                f.seek(0)
                with sr.AudioFile(f) as source:
                    audio = recognizer.record(source)
                    with open("audio/input.mp3", "wb") as mp3_file:
                        mp3_file.write(
                            audio.get_wav_data(convert_rate=16000, convert_width=2)
                        )
            # print("Audio saved as input.mp3")
        else:
            print("No speech detected")
        return base64_image


def main():
    full_analysis = []
    while True:
        final_image = get_input_file()
        user_prompt = get_prompt()
        print(user_prompt)
        pygame.mixer.init()
        pygame.mixer.music.unload()
        pygame.mixer.music.load("./audio/placeholder.mp3")
        pygame.mixer.music.play()
        analysis = analyze_image(full_analysis, final_image, user_prompt)
        print(analysis)
        # send_sms(analysis)
        play_audio(analysis)
        full_analysis = full_analysis + [{"role": "assistant", "content": analysis}]


if __name__ == "__main__":
    main()
