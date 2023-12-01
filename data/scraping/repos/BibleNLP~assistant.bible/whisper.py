"""
This module contains the WhisperAudioTranscription class which is used to
transcribe audio using OpenAI's Whisper API.
"""

import io
import os
import openai

from core.audio import AudioTranscriptionInterface
from custom_exceptions import AccessException

# pylint: disable= super-init-not-called, too-few-public-methods


class WhisperAudioTranscription(AudioTranscriptionInterface):
    """Interface for audio transcription technology and its use"""

    def __init__(
        self,  # pylint: disable=super-init-not-called
        key: str = os.getenv("OPENAI_API_KEY"),
    ) -> None:
        """Sets the API key and initializes the audio file object"""
        if key is None:
            raise AccessException(
                "OPENAI_API_KEY needs to be provided."
                + "Visit https://platform.openai.com/account/api-keys"
            )
        self.api_key = key
        self.api_object = openai
        self.api_object.api_key = key
        self.model = "whisper-1"
        self.audio_file = None

    def transcribe_audio(self, audio_data: bytes) -> str:
        """Generate transcription for the audio data"""
        self.audio_file = io.BytesIO()
        self.audio_file.write(audio_data)
        self.audio_file.seek(0)
        self.audio_file.name = "recorded_audio.wav"
        transcript = openai.Audio.transcribe(self.model, self.audio_file)

        return transcript["text"]
