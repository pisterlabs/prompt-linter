from fastapi import APIRouter
from pydantic import BaseModel
from ..adapters.cohere import CohereClient
from ..service.transcription import TranscriptionService
from ..lib.config import COHERE_API_KEY


class Transcription(BaseModel):
    raw_speech: str


cohere_client = CohereClient(COHERE_API_KEY)
transcription_service = TranscriptionService(cohere_client)

router = APIRouter()


@router.post("/transcription/translate", tags=["transcription"])
async def translate_transcription_route(transcription: Transcription):
    response = await transcription_service.translate_transcription(transcription.raw_speech)

    return {"latex": response}
