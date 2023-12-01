
import os
import json


import openai
from typing import Optional, List
from loguru import logger


from pydantic import BaseModel, Field

# 优先读取环境变量
API_KEY = os.environ.get('gpt_app_key')
api_type = os.environ.get('azure_api_type')
api_base = os.environ.get('azure_api_base')
api_version = os.environ.get('azure_api_version')
aip_engine = os.environ.get('azure_engine')


# 如果环境变量为空，则从 .env.json 文件中读取
if API_KEY is None or api_type is None or api_base is None or api_version is None:
    with open('./.env.json') as f:
        config = json.load(f)
        openai.api_key = API_KEY or config.get('gpt_app_key')
        openai.api_type = api_type or config.get('azure_api_type')
        openai.api_base = api_base or config.get('azure_api_base')
        openai.api_version = api_version or config.get('azure_api_version')
        aip_engine = aip_engine or config.get('azure_engine')


class MessageTurbo(BaseModel):
    model: Optional[str] = Field(default="gpt35-turbo", description="Model name")
    messages: Optional[List] = Field(default=None, description="Messages")
    max_tokens: Optional[int] = Field(default=2048, description="Stop sequence")
    temperature: Optional[float] = Field(default=0.5, description="Temperature")
    frequency_penalty: Optional[float] = Field(default=0.0, description="Frequency penalty")
    presence_penalty: Optional[float] = Field(default=0.0, description="Presence penalty")
    top_p: Optional[float] = Field(default=0.0, description="Top p")
    options: Optional[dict] = Field(default=None, description="Options")


class StreamMessageTurbo(BaseModel):
    model: Optional[str] = Field(default="gpt35-turbo", description="Model name")
    system_message: Optional[List] = Field(default=[], description="Messages")
    prompt: Optional[List] = Field(default=[], description="Messages")
    max_tokens: Optional[int] = Field(default=2048, description="Stop sequence")
    temperature: Optional[float] = Field(default=0.5, description="Temperature")
    top_p: Optional[float] = Field(default=0.0, description="Top p")
    options: Optional[dict] = Field(default=None, description="Options")


async def get_response_turbo(message):
    response = await openai.ChatCompletion.acreate(
        engine=aip_engine,
        messages=message.messages,
        temperature=message.temperature,
        max_tokens=message.max_tokens,
        frequency_penalty=message.frequency_penalty,
        presence_penalty=message.presence_penalty,
        stop=None)
    return response

#飞书专用请求
async def completions_turbo(message, retry):  # GPT3.5 turbo  feishu
    data = await get_response_turbo(message)
    # print(data)
    for i in range(retry):  # 重试，如果出错，重新请求一遍
        if data.get('error'):
            data = await get_response_turbo(message)
            if i == retry - 1:
                return '出错了，请稍后重试！！！'
        else:
            break
    return data["choices"][0]["message"]["content"]


#web 非stream流
async def web_completions_turbo(message):
    response = await openai.ChatCompletion.acreate(
        engine=aip_engine,
        messages=message.messages,
        temperature=message.temperature,
        max_tokens=message.max_tokens,
        stop=None)
    return response

#web stream流
async def stream_completions_turbo(stream_message: StreamMessageTurbo):
    response = await openai.ChatCompletion.acreate(
        engine=aip_engine,
        messages=stream_message.prompt,
        temperature=stream_message.temperature,
        max_tokens=stream_message.max_tokens,
        stream=True,
        stop=None)
    return response



