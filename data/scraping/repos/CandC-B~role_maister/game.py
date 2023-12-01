from fastapi import APIRouter
from utils.config import get_settings
from schemas.game_settings import GameSettings
from schemas.cohere_message import CohereMessage
from utils.game_utils import generate_new_game
from typing import List
from uuid import UUID
import cohere

game_router = APIRouter(tags=["Game"])

settings = get_settings()
co = cohere.Client(settings.COHERE_API_KEY)


@game_router.post("/")
async def create_new_game(game_settings: GameSettings):
    instruction = generate_new_game(game_settings)
    response = co.chat(
        message=instruction
    )

    return {"message": response.text}


@game_router.post("/master")
async def resume_game(chat_history: List[CohereMessage],message: str):
    chat_history = [chat.model_dump() for chat in chat_history]
    response = co.chat(
        chat_history=chat_history,
        message=message
    )

    return {"message": response.text}
