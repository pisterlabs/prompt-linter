from fastapi import APIRouter, WebSocket, BackgroundTasks
from app.schemas import message
from fastapi.responses import JSONResponse, StreamingResponse
import os
import openai
from dotenv import load_dotenv
from ..db import db_session as db
from ..db import Talk, TalkElement
from pydantic import BaseModel
import asyncio
from itertools import tee

load_dotenv()

router = APIRouter()
openai.api_key = os.getenv("OPENAI_API_KEY")

#会話を開始し、talk_idを返す
@router.get("/message/begin")
def begin_message():
    talk = Talk()
    db.add(talk)
    db.commit()
    return JSONResponse(status_code=200, content={'message': '会話を開始しました', 'talk_id': talk.id})


#バックグラウンドでレスポンスをDBに保存する処理
async def save_to_db(talk_id, response):
    all_content = []
    for chunk in response:
        if chunk:
            content = chunk['choices'][0]['delta'].get('content')
            if content:
                all_content.append(content)
    all = "".join(all_content)
    talkelement = TalkElement(role=0, content=all, order_id=0, talk_id=talk_id)
    db.add(talkelement)
    db.commit()

#messages配列を受け取って、chatGPTのストリーミングレスポンスを返す
@router.post("/message")
async def post_message(
    tasks: BackgroundTasks, item: message.Message
    ):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {'role': m.role, 'content': m.content} for m in item.messages
        ],
        stream=True
    )
    #generatorを2つに分ける
    gen1, gen2 = tee(response, 2)
    #バックグラウンドでDBに保存する
    tasks.add_task(save_to_db, item.talk_id, gen1)

    def iterfile():
        for chunk in gen2:
            if chunk:
                content = chunk['choices'][0]['delta'].get('content')
                if content:
                    yield content
    return StreamingResponse(iterfile(), media_type="text/plain")