from io import FileIO
from typing import Dict

try:
    from typing.io import BinaryIO
except ImportError:
    from typing import BinaryIO

from polly.client.openai import OpenAIClient

from pydub import AudioSegment


class Whisper:

    SUPPORTED_TYPES = [
        'm4a', 'mp3', 'webm', 'mp4', 'mpga', 'wav', 'mpeg'
    ]

    def __init__(self, client: OpenAIClient):
        self.client = client.openai_api
        self.model_name = client.openai_model.get(
            client.WHISPER_KEY
        )

    def load_audio(self, filepath: str) -> BinaryIO:
        ext = filepath.split('.')[-1]
        if ext not in self.SUPPORTED_TYPES:
            raise ValueError(f'Unsupported file type : {ext}')

        file = BinaryIO(FileIO(filepath, 'rb'))
        return file

    @staticmethod
    async def convert_audio(fr: str, to: str) -> None:
        AudioSegment.from_file(fr).export(to, format="mp3")

    def transcribe(self, audio_file: BinaryIO) -> Dict:
        response = self.client.Audio.transcribe(
            model=self.model_name,
            file=audio_file,
        )
        return response
