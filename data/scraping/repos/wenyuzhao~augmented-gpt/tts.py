from typing import Literal, Optional

import openai
from pathlib import Path

Voice = Literal["alloy", "echo", "fable", "onyx", "nova", "shimmer"]


class TextToSpeech:
    def __init__(
        self,
        api_key: str,
        model: Literal["tts-1", "tts-1-hd"] = "tts-1",
        voice: Voice = "alloy",
    ):
        self.client = openai.AsyncOpenAI(api_key=api_key)
        self.voice: Voice = voice
        self.model = model

    async def speak(
        self,
        text: str,
        output: str | Path,
        voice: Optional[Voice] = None,
    ):
        _voice: Voice = voice or self.voice
        response = await self.client.audio.speech.create(
            model=self.model, voice=_voice, input=text
        )
        response.stream_to_file(output)

    def speak_sync(
        self,
        text: str,
        output: str | Path,
        voice: Optional[Voice] = None,
    ):
        from . import block_on

        block_on(self.speak(text, output, voice))


__all__ = ["TextToSpeech", "Voice"]
