from output.output_audio.tts.tts_interface import TTSInterface
from config import config
from openai import OpenAI


class TTSOpenAi(TTSInterface):
    def __init__(self) -> None:
        self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        self.voice = "shimmer"
        self.model = "tts-1"

    def text_to_speech(self, text: str) -> str:
        response = self.client.audio.speech.create(
            model=self.model,
            voice=self.voice,
            input=text,
        )

        filename = f"{config.TEMP_AUDIO_FOLDER}/output.mp3"
        response.stream_to_file(filename)

        return filename
