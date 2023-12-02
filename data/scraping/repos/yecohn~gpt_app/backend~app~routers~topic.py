from fastapi import APIRouter, Depends, HTTPException, status, Security
from fastapi import UploadFile, File, Depends
from backend.db.sql.sql_connector import access_sql
from backend.db.sql.tables import User
from backend.db.mongo.mongo_connector import access_mongo
from backend.engine.gpt import GPTClient
from backend.engine.speech_to_text import STT
from backend.engine.text_to_speech import GCPTTS
from backend.app.users.user import UserInfo
import openai
from google.cloud import speech
from ..oauth2 import get_current_user
from backend.app.models import MessageChat
from datetime import datetime

router = APIRouter()


@router.get("/topics/", status_code=200)
async def topic_list(
    mongo_db=Depends(access_mongo),
):
    """_summary_

    Args:
        mongo_db (_type_, optional): _description_. Defaults to Depends(access_mongo).

    Returns:
        _type_: _description_
    """
    topics = mongo_db.find_all_but(
        collection_name="topics", projection={"transcript": 0}
    )
    res = []
    for topic in topics:
        res.append(
            {
                "topic_id": topic["topic_id"],
                "name": topic["name"],
                "description": topic["description"],
                "link": topic["link"],
            }
        )
    return res


@router.get("/topics/{topic_id}", status_code=200)
async def triggerTopic(
    topic_id: int,
    user_id: int,
    mongo_db=Depends(access_mongo),
):
    """_summary_

    Args:
        topic_id (int): _description_
        mongo_db (_type_, optional): _description_. Defaults to Depends(access_mongo).

    Returns:
        _type_: _description_
    """
    gpt = GPTClient()
    transcript = mongo_db.find(
        collection_name="topics", query={"topic_id": topic_id}
    ).get("transcript")
    topic_prompt = (
        mongo_db.find(collection_name="metadata")
        .get("GPT_metadata")
        .get("topic_prompt")
    )
    answer = gpt.ask_gpt_about_topic(transcript, topic_prompt)
    answer_json = {
        "user": {"id": 0, "name": "ai"},
        "text": answer,
        "createdAt": datetime.now(),
    }

    mongo_db.push(
        "chats",
        {"user_id": user_id},
        {"$push": {"messages": answer_json}},
    )

    # _ = tts.generate_speech(answer)
    return {"ok": True}
