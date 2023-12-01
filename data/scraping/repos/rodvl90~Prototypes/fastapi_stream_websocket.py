import asyncio
import json
import os
from pprint import pprint
from typing import List

from pydantic import BaseModel

import openai
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from openai import ChatCompletion

# Set up OpenAI
openai.api_key = "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"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Assuming your Svelte app runs on this host and port
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH", "CONNECT", "TRACE", "WebSocket"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    async def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def send_chat_message(self, message: str):
        history = [{"role": "system", "content": "you are a helpful ai!"},{"role": "user", "content": "Hi!My name is vlad!Can you write a 20 verse poem?"}]
        res = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=history,
            temperature=0.5,
            max_tokens=500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0.6,
            stream=True
        )
        for connection in self.active_connections:
                for chunk in res:
                    chunk_message = chunk['choices'][0]['delta']
                    await connection.send_text(chunk_message["content"])
        

manager = ConnectionManager()
@app.websocket("/ws_test")
async def ws(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await manager.send_chat_message(websocket)
    except WebSocketDisconnect:
        await manager.disconnect(websocket)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        messages = json.loads(data)
        pprint(messages)
        chat_model = "gpt-4"  # Use an appropriate model
        chat_log = [{"role": message["role"], "content": message["content"]} for message in messages]

        async for message in ChatCompletion.create(model=chat_model, messages=messages, stream=True):
            await websocket.send_text(message['choices'][0]['delta']['content'])
