import os
import sys
from pprint import pprint
from typing import Dict, List

from pydantic import BaseModel

import openai
from fastapi import FastAPI, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

openai.api_key = "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"
class OpenaiChatMessage(BaseModel):
    role: str
    content: str
class OpenaiChatMessagesRequest(BaseModel):
    messages: List[OpenaiChatMessage]

# Parameters for OpenAI
openai_model = "gpt-3.5-turbo"
max_responses = 1
temperature = 0.7
max_tokens = 512

# Defining the FastAPI app and metadata
app = FastAPI(
    title="Streaming API",
    description="""### API specifications\n
To test out the Streaming API `chat`
              """,
    version=1.0,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Assuming your Svelte app runs on this host and port
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH", "CONNECT", "TRACE", "WebSocket"],
    allow_headers=["*"],
)
# Defining error in case of 503 from OpenAI
error503 = "OpenAI server is busy, try again later"


def get_response_openai(messages):
    try:
        messages = messages.dict().get("messages")
        pprint(messages)
        response = openai.ChatCompletion.create(
            model=openai_model,
            temperature=temperature,
            max_tokens=max_tokens,
            n=max_responses,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            messages=messages,
            stream=True,
        )
    except Exception as e:
        print("Error in response from openAI:", str(e))
        raise HTTPException(503, error503)
    try:
        for chunk in response:
            current_content = chunk["choices"][0]["delta"].get("content", "")
            print(current_content)
            yield current_content
    except Exception as e:
        print("OpenAI Response (Streaming) Error: " + str(e))
        raise HTTPException(503, error503)


@app.post("/chat",tags=["chat"])
def chat(messages: OpenaiChatMessagesRequest):
    return StreamingResponse(get_response_openai(messages), media_type="text/event-stream")