import keyboard
import cv2
import sounddevice as sd
import numpy as np
import wavio
import base64
import requests
from openai import OpenAI
from playsound import playsound
import logging
import threading
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)

openai_api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=openai_api_key,
    # default is 2
    max_retries=0,
)

def listen_for_keypress(key):
    keyboard.wait(key)

def capture_image(filename):
    cap = cv2.VideoCapture(-0)
    cap.set(3,640)
    cap.set(3,480)  # 0 is typically the default camera
    ret, frame = cap.read()
    while(True):
        success, img = cap.read()

        cv2.imshow('frame',img)

        if cv2.waitKey(1) & 0xFF == ord('p'):
            cv2.imwrite(filename, img)
            break
    cap.release()
    cv2.destroyAllWindows()

class AudioRecorder:
    def __init__(self, filename, fs=44100):
        self.filename = filename
        self.fs = fs
        self.recording = np.array([])
        self.is_recording = False
        self.thread = threading.Thread(target=self.record_background)

    def record_background(self):
        with sd.InputStream(samplerate=self.fs, channels=1, dtype='int16') as stream:
            self.is_recording = True
            while self.is_recording:
                data, _ = stream.read(1024)  # Read chunks of 1024 frames
                self.recording = np.append(self.recording, data)

    def normalize_audio(self, audio):
        """
        Normalize the audio data to be within the range -1.0 to 1.0
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio

    def start(self):
        self.thread.start()

    def stop(self):
        self.is_recording = False
        self.thread.join()
        wavio.write(self.filename, self.normalize_audio(self.recording.reshape(-1, 2)), self.fs/2, sampwidth=2)

def record_audio_on_keypress(filename, key='r'):
    """
    Records audio until the specified key is pressed.

    Args:
    filename (str): The name of the file to save the recording.
    key (str, optional): Key to stop recording. Defaults to 'r'.
    """
    recorder = AudioRecorder(filename)
    recorder.start()
    print(f"Recording... Press '{key}' to stop.")

    keyboard.wait(key)  # Wait for key press
    recorder.stop()
    print("Recording stopped.")

def transcribe_audio(file_path):
    with open(file_path, 'rb') as audio_file:
        response = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1"
        )
    return response

def ask_gpt4(question):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "user", "content": question}
        ]
    )
    return response.choices[0].message.content

def text_to_speech(text, filename):
    response = client.audio.speech.create(
        model="tts-1",
        voice="shimmer",
        input=text
    )
    with open(filename, "wb") as f:
        f.write(response.content)
    playsound(filename)

def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8');

def ask_gpt4_with_image(image_path, text, api_key):
    base64_image = encode_image(image_path)

    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
    }

    payload = {
      "model": "gpt-4-vision-preview",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": text
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
      "max_tokens": 300
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    return response.json()

def generate_datecoded_filename(prefix, extension):
    datecode = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"data/{prefix}_{datecode}.{extension}"

def main():
    # File paths
    audio_file_path = generate_datecoded_filename("audio", "wav")
    image_file_path = generate_datecoded_filename("image", "jpg")

    logging.info("Starting the application. Press 'r' to start. 'p' to take a picture once the camera is on.")

    record_audio_on_keypress(audio_file_path, 'r')

    logging.info('Loading camera...');

    # Wait for 'p' to capture an image
    capture_image(image_file_path)
    logging.info(f"Picture taken and saved as {image_file_path}")

    logging.info('Processing audio and image...')
    # Transcribe audio
    transcription = transcribe_audio(audio_file_path)
    logging.info(f"Transcription: {transcription}")

    # Interact with GPT-4
    gpt_response = ask_gpt4_with_image(image_file_path, transcription.text, openai_api_key)
    gpt_response_text = gpt_response["choices"][0]["message"]["content"]
    logging.info(f"GPT-4 Response: {gpt_response_text}")


    # Convert response to speech
    text_to_speech(gpt_response_text, generate_datecoded_filename("response", "mp3"))
    logging.info("Response has been played back as speech.")

if __name__ == "__main__":
    main()