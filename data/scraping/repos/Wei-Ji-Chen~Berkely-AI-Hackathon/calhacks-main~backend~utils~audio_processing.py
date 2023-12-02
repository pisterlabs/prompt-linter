import base64
import os
from backend.data import DATA_DIR
import openai
from datetime import datetime
import azure.cognitiveservices.speech as speechsdk
from backend.config import SPEECH_KEY, SPEECH_REGION


async def speech_to_text(path: str):
    """
    audio: base-64 string that is user input
    file_type: file type
    """
    audio_file = open(path, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript["text"]

async def save_as_file(audio: str):
    """
    Saves audio as file, returns path
    """
    decoded = base64.b64decode(audio)
    time = datetime.now()
    path = os.path.join(DATA_DIR, str(time) + ".webm")
    audio_bytes = bytearray(decoded)
    with open(path, "wb") as bytes_file:
        bytes_file.write(audio_bytes)

    return str(path)

def text_to_speech(text: str, path: str):
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    audio_config = speechsdk.audio.AudioOutputConfig(filename=path)

    # The language of the voice that speaks.
    speech_config.speech_synthesis_voice_name='en-US-JennyNeural'
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=audio_config)

    speech_synthesis_result = speech_synthesizer.speak_text_async(text).get()

    with open(path, "rb") as file:
        data = file.read()
        encoded = base64.b64encode(data)
        return encoded.decode("utf-8")
