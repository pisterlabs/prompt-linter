from fastapi import APIRouter
from ..models import GetOutlineRequest
from services import openai_service
from ..utils import digest_outline

router = APIRouter()

@router.post("/getoutline", response_model=dict, tags=["Outline"], summary="Get topic headings based on the given topic and instructions.")
async def get_headings(request: GetOutlineRequest):
    """
    Given a topic and custom instructions, this endpoint interacts with the OpenAI API to get a list of relevant headings.
    """
    topic = request.topic
    custom_instructions = request.custom_instructions

    response = openai_service.get_outline_sync(topic, custom_instructions)
    items = digest_outline(response)

    return {"items": items}
