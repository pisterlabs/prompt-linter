from typing import *
from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session
from db.session import *
from crud import crud_item
from core.config import settings
import sys
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi.encoders import jsonable_encoder
from schema.item import ItemBase, ItemScheme, ChatMessage
from core.log_config import base_logger
import os
from crud.crud_item import *
import requests
import openai
from openai import AsyncOpenAI
import re
from ArtificalIntelligence.witAI import WitAI, witai, WitResponse
from ArtificalIntelligence.summarize import get_summarize_object

router = APIRouter()


@router.post("/db")
async def db_test(request: Request, data: ItemBase, db: AsyncSession = Depends(get_db)):
    # data = jsonable_encoder(data)
    result = await crud_item.insert(db=db, data=data)
    base_logger.info(msg=f"{dict(request)}")
    return result


@router.post("/test/redis")
async def test_redis(request: Request, msg: str, redis=Depends(get_redis)):
    value = await redis.get(msg)
    redis_response = await redis.set("test", "hello")
    return {"result": value}


@router.post('/chat')
async def get_message(request: Request, chatmessage: ChatMessage, db: AsyncSession = Depends(get_db),
                      redis=Depends(get_redis)):
    # get response from witai
    try:
        wit_response = await witai.get_response(chatmessage.messages)
        wit_response_json = await wit_response.json()
        wit: WitResponse = WitResponse(response=wit_response_json, status=wit_response.status)
        wit_intent = wit.get_intent()
        wit_confidence = wit.get_confidence()
        if wit.status == 200 and wit_confidence > 0.95:
            wit_intent = wit.get_intent()
            # 만약 intent 가 redis 에 있다면 바로 반환 GPT 까지 갈필요 X
            value = await redis.get(wit_intent)
            if value is not None:
                # print("cache value : ", value.decode('utf-8'))
                return {"result": value}

        gpt = AsyncOpenAI(api_key=settings._GPT_API_KEY)
        # print(chatmessage)
        response = await gpt.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                 "content": "너는 GPT가 아닌 Chatbot이야 사용자의 질문에 적절한 대답을 해줄 system이야 그리고 너의 이름은 콩돌이 Chatbot이야"
                            "답변은 300자내로 끊어서 대답해주면 돼"},
                {"role": "assistant",
                 "content": "너는 <Chatbot assistant>야 사용자의 질문에 적절한 대답을 해줄 의무가있는 assistant야"
                 },
                {"role": "user", "content": chatmessage.messages},

            ],
            temperature=0.2
        )
        response_str = response.choices[0].message.content
        response_str = re.sub("gpt | GPT | OpenAI | openai | chatgpt", "", response_str)
        if wit_confidence > 0.95:
            await redis.set(wit_intent, response_str)
        return {"result": response_str}
    except Exception as e:
        print(e)
        return {"result": "retry next time"}

import json
@router.post("/summarize-video")
async def get_summarize(request : Request, video_id : str,summarize_handler = Depends(get_summarize_object) ):
    print(dict(request))
    response = await summarize_handler.get_summarize(video_id=video_id)
    base_logger.info(response)
    return response
