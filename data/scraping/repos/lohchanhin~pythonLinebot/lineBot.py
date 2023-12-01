from fastapi import FastAPI, Request, status, HTTPException
from linebot import LineBotApi, WebhookParser
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
import os
import openai as ai
import requests
from typing import Dict, List

# 获取 LINE 密钥
channel_access_token = os.getenv('CHANNEL_ACCESS_TOKEN')
channel_secret = os.getenv('CHANNEL_SECRET')

# 创建 LINE 客户端
line_bot_api = LineBotApi(channel_access_token)
parser = WebhookParser(channel_secret)

app = FastAPI()

# 存储用户会话的对象
user_conversations: Dict[str, List[Dict[str, str]]] = {}


@app.post("/callback")
async def callback(request: Request):
    body = await request.body()
    signature = request.headers.get('X-Line-Signature')

    try:
        events = parser.parse(body.decode('utf-8'), signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST)

    for event in events:
        if isinstance(event, MessageEvent):
            handle_message(event)

    return "OK"


def handle_message(event: MessageEvent):
    # 如果消息类型不是文本，则忽略
    if not isinstance(event.message, TextMessage):
        return

    # 进行自然语言处理并回复用户
    text = event.message.text
    user_id = event.source.user_id

    # 如果不存在该用户的对话，为其创建一个
    if user_id not in user_conversations:
        user_conversations[user_id] = [
            {"role": "system", "content": '你是人工智能助理'}
        ]

    # 将用户消息添加到会话中
    user_conversations[user_id].append({"role": "user", "content": text + '回答字数限制在1000以内'})

    # 如果会话长度超过 4 条消息，则删除最早的一条
    if len(user_conversations[user_id]) > 4:
        user_conversations[user_id].pop(0)

    # 获取 OpenAI API 密钥
    openai_api_key = os.getenv('OPENAI_API_KEY')

    # 使用 OpenAI API 获取回复
    ai.api_key = openai_api_key
    openai_response = ai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=user_conversations[user_id],
    )

    # 获取助手回复的文本
    assistant_reply = openai_response['choices'][0]['message']['content']

    # 将助手回复添加到对话中
    user_conversations[user_id].append({"role": "assistant", "content": assistant_reply})

    # 使用 LINE API 回复用户
    line_bot_api.reply_message(event.reply_token, TextSendMessage(text=assistant_reply))
