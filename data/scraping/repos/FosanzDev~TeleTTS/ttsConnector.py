from openai import AsyncOpenAI
from pathlib import Path
from datetime import datetime

class TTSConnector():

    def __init__(self, apiKey):
        self.client = AsyncOpenAI(api_key=apiKey)


    async def synth(self, text: str, chat_id) -> Path:
        file = Path(__file__).parent / f"tts_{chat_id}_{datetime.now().strftime('%Y%m%d%H%M%S')}.ogg"
        response = await self.client.audio.speech.create(
            model="tts-1",
            voice="echo",
            input=text
        )
        response.stream_to_file(file)
        return file