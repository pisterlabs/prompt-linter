import os
import json
from typing import Dict

from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

SYSTEM_MESSAGE = """You are an environmental scientist that will generate a detailed description of a region based on the prompt starting with 'request: ' and the locations provided as 'contextual data'.
Your description should sound like it was spoken by someone with personal knowledge of the region. 
Format any names or places you get as bold text in Markdown.
Do not mention who you are or the Overpass API, just give the description of the place."""

# take environment variables from .env.
load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")


async def get_openai_api_response(data, chat) -> dict[str, str] | str:
    request = f"request: {chat.text}"
    context = f"""contextual data: {json.dumps(data.dict())}"""

    chat = ChatOpenAI(temperature=.7, openai_api_key=openai_api_key)

    try:
        chat_response = chat(
            [
                SystemMessage(content=SYSTEM_MESSAGE),
                AIMessage(content=context),
                HumanMessage(content=request)
            ]
        )
    except Exception as e:
        return {"error": str(e)}

    description = chat_response.content

    return description
