import torch
import openai
import dotenv
import os
import time
import json

from fastapi import APIRouter

from typing import Union
from pydantic import BaseModel

from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    SystemMessage,
    AIMessage,
    HumanMessage,
)
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

import asyncio
import os
from typing import AsyncIterable, Awaitable
from fastapi.responses import StreamingResponse
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
)

from databases import Database

## 데베 설정
DATABASE_URL = "mysql+pymysql://root:1234@127.0.0.1:3306/place_classification"
database = Database(DATABASE_URL)


## Open AI 설정
dotenv_file = dotenv.find_dotenv()
dotenv.load_dotenv(dotenv_file)

key = os.environ["OPENAI_API_KEY"]
openai.api_key = key

from openai import OpenAI
client = OpenAI()

## Fast api 연결
router = APIRouter(
    prefix="/chatbot",
)


## 프롬프트 데이터 가져오기
json_path='chat_data.json'

def connect_json():
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


######################################################
### 퀴즈 생성

class quiz_gen(BaseModel):
    text: str

@router.post("/quiz_gen")
def image_def(input: quiz_gen):
        print(input.text)

        ## json 연결
        json_data=connect_json()["quiz_gen"]

        messages = [
            {"role": "system", "content": json_data["system"]},
            {"role": "user", "content": json_data["input"][0]},
            {"role": "assistant", "content": json_data["output"][0]},
            {"role": "user", "content": json_data["input"][1]},
            {"role": "assistant", "content": json_data["output"][1]},
            {"role": "user", "content": json_data["input"][2]},
            {"role": "assistant", "content": json_data["output"][2]},
            {"role": "user", "content": input.text}
        ]
        try:
            chat_completion = client.chat.completions.create(  ## gpt 오브젝트 생성후 메세지 전달
                model="gpt-4",
                messages=messages,
                temperature=1,
                max_tokens=1000
            )
            result = chat_completion.choices[0].message.content
            output = json.loads(result)
        except Exception as e: #나중에 더미데이터 넣기
            output='''"제목: \"크리스마스 카드를 보내는 미국의 문화 이야기\"|\n\n(카메라에게 인사하며)|\n\"안녕. 오늘은 작년 크리스마스 때 벌어진 한 이야기를 들려주고 싶어.\" |\n\n내용:|\n(카메라에게 인사하며)|\n\"그때 내가 미국에서 알게 된 친구에게 크리스마스 카드를 안 보냈거든.\" |\n\"그랬더니 그 친구가 진짜로 굉장히 서운해했어.\" |\n\n(카메라에게 인사하며)|\n\"미국에서는 크리스마스 카드를 주고 받는 게 일상적인 문화라고 하더라고.\" |\n\"그래서 그 친구 가 왜 그렇게 서운해 하는지 이해가 갔어.\" |\n\n결론:|\n\" 미국인 친구가 있으면 꼭 편지를 써줘야 해\" |\n\n(카메라에게 웃으며 손을 흔들며)|\n\"다음에 또 재미있는 이야기로 찾아올게. 바이바이~.\"\n\n"'''

            
        print(output)
        return output

######################################################

class AsyncStringIterator:
    def __init__(self, string):
        self.string = string
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index < len(self.string):
            result = self.string[self.index]
            self.index += 1
            await asyncio.sleep(0.1)  # 비동기 작업을 시뮬레이션
            return result
        else:
            raise StopAsyncIteration

        
#### 비동기 스트리밍 통신
async def send_message(text: str) -> AsyncIterable[str]:
    final_token=""
    error_event = asyncio.Event()

    callback = AsyncIteratorCallbackHandler()
    model = ChatOpenAI(
        model_name="gpt-4",
        streaming=True,
        verbose=True,
        callbacks=[callback],
    )

    async def wrap_done(fn: Awaitable, event: asyncio.Event):
        nonlocal final_token
        try:
            await fn
        except Exception as e:
            final_token = "Error"
            print(final_token)

            

    ## json 연결
    json_data=connect_json()["make_script"]

    task = asyncio.create_task(wrap_done(
        model.agenerate(messages=[[SystemMessage(content=json_data['system']),HumanMessage(content=json_data['input'][0]),AIMessage(content=json_data['output'][0]),
                                   HumanMessage(content=json_data['input'][1]),AIMessage(content=json_data['output'][1]),
                                HumanMessage(content=json_data['input'][2]),AIMessage(content=json_data['output'][2]),
                                HumanMessage(content=json_data['input'][3]),AIMessage(content=json_data['output'][3]),
                                  HumanMessage(content=text)]]),
        callback.done),
    )

    n=0

    async for token in callback.aiter():
        print(n, end=" ")
        n += 1
        print(token)
        yield f"data: {token}\n\n"

    await task



    if final_token.startswith("Error"):
        async_string = AsyncStringIterator("에러입니다.")
        async for char in async_string:
            yield f"data: {char}\n\n"
        # yield f"data: {final_token}\n\n"

    print(4)
    
class StreamRequest(BaseModel):
    text: str

@router.post("/make_script")
def stream(body: StreamRequest):
    print(body.text)
    return StreamingResponse(send_message(body.text), media_type="text/event-stream")

######################################################
### 이미지 분류 통신
class Image_connect(BaseModel):
    text: str


@router.post("/image_connect")
async def image_def(input: Image_connect):
    print(input.text)

    ## json 연결
    json_data=connect_json()["image_connect"]

    messages = [
        {"role": "system", "content": json_data["system"]},
        {"role": "user", "content": json_data["input"][0]},
        {"role": "assistant", "content": json_data["output"][0]},
        {"role": "user", "content": json_data["input"][1]},
        {"role": "assistant", "content": json_data["output"][1]},
        {"role": "user", "content": input.text}
    ]
    try:
        chat_completion = client.chat.completions.create(  ## gpt 오브젝트 생성후 메세지 전달
            model="gpt-4",
            messages=messages,
            temperature=1,
            max_tokens=1000
        )

        result = chat_completion.choices[0].message.content

        query = "INSERT INTO place_table (Q, A) VALUES (:Q, :A)"
        values = {"Q": input.text, "A": result}
        await database.execute(query=query, values=values)
    except Exception as e: #나중에 더미데이터 넣기
        result="gpt 에러"

    ## 시연전용
    result ="home,winter"

    print("분류결과 : "+result)
    return {"place" : result}
######################################################


######################################################
### 음악 분류 통신
class Sound_connect(BaseModel):
    text: str


@router.post("/music_connect")
async def Sound_def(input: Sound_connect):
    print(input.text)
    
    ## json 연결
    json_data=connect_json()["music_connect"]

    messages = [
        {"role": "system", "content": json_data['system']},
        {"role": "user", "content": json_data['input'][0]},
        {"role": "assistant", "content": json_data['output'][0]},
        {"role": "user", "content": input.text}
    ]

    try:
        chat_completion = client.chat.completions.create(  ## gpt 오브젝트 생성후 메세지 전달
            model="gpt-4",
            messages=messages,
            temperature=1,
            max_tokens=1000
        )
        
        result = chat_completion.choices[0].message.content


        query = "INSERT INTO bgmusic_table (Q, A) VALUES (:Q, :A)"
        values = {"Q": input.text, "A": result}
        await database.execute(query=query, values=values)
    except Exception as e: #나중에 더미데이터 넣기
        result="gpt 에러"

    print("분류결과 : "+result)
    result="Peaceful"
    return {"mood" : result}
######################################################

@router.on_event("startup")
async def startup():
    await database.connect()

@router.on_event("shutdown")
async def shutdown():
    await database.disconnect()