import os
import openai
from fastapi import FastAPI
from sse_starlette.sse import EventSourceResponse
from pydantic import BaseModel
import uvicorn
import json

openai.api_key = os.getenv("OPENAI_API_KEY")

class AskRequest(BaseModel):
    query: str


async def ask_llm_stream(query: str):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        stream=True,  # SSEを使うための設定
        messages=[
            {
                "role": "user",
                "content": f"{query}"
            }
        ],
    )

    for item in response:
        try:
            content = item['choices'][0]['delta']['content']
        except:
            content = ""
        # dict型で返すことでよしなに変換してくれる
        yield {"data": content}
    yield {"data": "[DONE]"}


app = FastAPI()


@app.post("/streaming/ask")
async def ask_stream(ask_req: AskRequest) -> EventSourceResponse:
    # イテラブルオブジェクトを受け取る
    return EventSourceResponse(ask_llm_stream(ask_req.query))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
