import logging
import os
from collections import OrderedDict
from typing import Iterator

import openai
import sounddevice as sd
from dotenv import load_dotenv
from elevenlabs import generate, set_api_key, stream, voices
from scipy.io.wavfile import write

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
debug = os.getenv("DEBUG", False)

if debug:
    logging.basicConfig(level=logging.DEBUG)


def record_audio() -> None:
    """Records audio from device"""
    logging.debug("Listening")
    fs = 44100  # Sample rate
    seconds = 3  # Duration of recording

    myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=2)
    sd.wait()  # Wait until recording is finished
    write("output.wav", fs, myrecording)  # Save as WAV file


def speech2text() -> str:
    """Converts audio file into text

    Returns:
        str: audio text transcript
    """
    audio_file = open("output.wav", "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)["text"]
    logging.debug(transcript)
    return transcript


def get_answer(messages: list[dict]) -> str:
    """Returns answer based on the conversation so far

    Args:
        messages (list[dict]): A list of messages comprising the conversation so far.

    Returns:
        str: Genrated answer
    """
    logging.debug("Getting answer")
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Tu es un assistant un peu fou. Tu as beacoup d'humour. Tes réponses sont généralement courtes, seulement quelques phrases.",
            },
            *messages,
        ],
    )

    answer = completion.choices[0].message["content"]
    logging.debug(answer)
    return answer


def text2speech(text: str, voice: str) -> Iterator[bytes]:
    """Transforms text into audio

    Args:
        text (str): text to transform
        voice (str): elevenlabs voice id

    Returns:
        Iterator[bytes]: generated audio
    """
    logging.debug("Reading answer")
    audio = generate(
        api_key=elevenlabs_api_key,
        text=text,
        voice=voice,
        model="eleven_multilingual_v1",
        stream=True,
    )

    return audio


def get_voices(custom: bool = True) -> OrderedDict:
    """Get available elevenlabs voices

    Args:
        custom (bool): whether to return custom voices or only the default ones.

    Returns:
        OrderedDict: available voices
    """
    if custom:
        set_api_key(elevenlabs_api_key)
    available_voices = voices()

    def format_voices(categories):
        voices = []
        for voice in available_voices.voices:
            if voice.category in categories:
                labels = voice.labels if voice.labels else {}
                use_case = labels.pop("use case", "")
                description = "" if labels is None else " ".join(list(labels.values()))
                voices.append(
                    (
                        voice.name,
                        {
                            "id": voice.voice_id,
                            "description": description,
                            "use case": use_case,
                        },
                    )
                )
        return voices

    cloned_voices = format_voices(["cloned", "generated"])
    premade_voices = format_voices(["premade"])

    chosen_voices = OrderedDict([*cloned_voices, *premade_voices])
    return chosen_voices
