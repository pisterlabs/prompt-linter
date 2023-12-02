# This file is used to handle the azure ai chatbot
import uuid
from queue import Queue

from cachetools import TTLCache
from flask import Blueprint, request, Response
from langchain import ConversationChain, LLMChain
from langchain.chat_models import AzureChatOpenAI
import os
import threading

from langchain.memory import ConversationBufferMemory

from app.azureai.handler.stream_handler import QueueCallbackHandler
from app.azureai.langchain.chain.prompts import message_prompt

azure_ai = Blueprint('azure_ai', __name__)
os.environ['OPENAI_API_TYPE'] = "azure"
os.environ['OPENAI_API_BASE'] = 'https://jiawei.openai.azure.com/'
# os.environ['OPENAI_API_VERSION'] = "2022-12-01"
os.environ['OPENAI_API_VERSION'] = "2023-03-15-preview"

ai = AzureChatOpenAI(
    deployment_name='gpt-35-turbo',
    model_name='gpt-35-turbo',
    streaming=True,
    temperature=0.9,
)

sessions = {}

ttl_cache = TTLCache(maxsize=10, ttl=60 * 60 * 24)


def llm_thread(q, input):
    if "id" in input:
        uid = input["id"]
        if uid in sessions:
            chain = sessions[uid]
        else:
            chain = ConversationChain(
                llm=ai,
                memory=ConversationBufferMemory()
            )
            sessions[uid] = chain
    else:
        chain = LLMChain(
            llm=ai,
            prompt=message_prompt,
        )
    chain.run(input["message"], callbacks=[QueueCallbackHandler(q)])


def stream(q, input):
    threading.Thread(target=llm_thread, args=(q, input)).start()
    while True:
        message = q.get()
        message = message.replace('\n', '&#92n;')
        if message != "":
            yield f'data: %s\n\n' % message


@azure_ai.route('/chat', methods=['GET', 'POST'])
def chat():
    q = Queue()
    if request.method == 'POST':

        input = request.get_json()
        verify_token = input.get("verifyToken", None)
        if verify_token in ttl_cache:
            # asyncio.run(dochat(input_text, q))
            return Response(stream(q, input), mimetype='text/event-stream')
        else:
            return {
                "code": 200,
                "success": False,
                "data": "verify token error"
            }
    else:
        return Response(None, mimetype='text/event-stream')


@azure_ai.route('/clear')
def clear():
    uid = request.args.get("id")
    if uid in sessions:
        del sessions[uid]
    return {
        "code": 200,
        "success": True,
        "data": "success"
    }


@azure_ai.route('/verify', methods=['GET', 'POST'])
def verify():
    param = request.get_json()
    verify_code = param["verifyCode"]
    # 生成token放入cachetools
    verify_token = uuid.uuid4().hex
    ttl_cache[verify_token] = verify_code
    if verify_code == os.getenv("VERIFY_CODE", "qpzm"):
        return {
            "code": 200,
            "success": True,
            "data": verify_token
        }
    return {
        "code": 200,
        "success": False,
        "data": "认证失败"
    }
