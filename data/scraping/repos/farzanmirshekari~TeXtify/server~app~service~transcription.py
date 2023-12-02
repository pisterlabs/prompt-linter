from ..adapters.cohere import CohereClient


class TranscriptionService:
    def __init__(self, cohere_client: CohereClient):
        self.cohere_client = cohere_client

    async def translate_transcription(self, raw_speech: str) -> str:
        response = await self.cohere_client.generate_latex(raw_speech)
        print(f"received response {response}")
        if response and response.generations:
            return response.generations[0].text
        else:
            return ""
