import tempfile
from io import BytesIO
from typing import Literal

from src.config import openai


def text_to_speech(
    input_text: str, voice: Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
) -> BytesIO:
    """
    Synthesize text to speech
    :return: binary reponse
    """

    binary_response = openai.audio.speech.create(
        model="tts-1",
        voice=voice,
        input=input_text,
    )

    response_file = tempfile.NamedTemporaryFile(mode="w+", suffix=".mp3", delete=False)
    binary_response.stream_to_file(response_file.name)
    return response_file
