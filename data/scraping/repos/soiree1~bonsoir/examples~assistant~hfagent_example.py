import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import random
import yaml

from transformers.tools import OpenAiAgent, Tool
from itllib import Itl

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
SECRETS_PATH = "./secrets"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", None)


itl = Itl()
itl.apply_config(CONFIG_PATH, SECRETS_PATH)
itl.start_thread()

executor = ThreadPoolExecutor(max_workers=1)
agent = OpenAiAgent(model="gpt-4-0613", api_key=OPENAI_API_KEY)
agent.toolbox.clear()

class Greeter(Tool):
    def __init__(self):
        super().__init__()
        self.name = 'greeter'
        self.description = (
            "This is a tool that greets the user. "
            "It takes an input named `username` which should be a "
            "string representation of the user's name. It "
            "returns a text that contains the greeting."
        )
    def __call__(self, username: str):
        return f'Hello, {username}!'

agent.toolbox['greeter'] = Greeter()

loop = asyncio.get_event_loop()

@itl.ondata("one-off-requests")
async def handle_one_off_request(request):
    response = await loop.run_in_executor(executor, agent.run, request)
    await itl.stream_send("responses", response)


@itl.ondata("chat-requests")
async def handle_chat_request(request):
    response = await loop.run_in_executor(executor, agent.chat, request)
    await itl.stream_send("responses", response)


@itl.ondata("reset-chat")
async def handle_reset_chat(*args, **kwargs):
    agent.prepare_for_new_chat()
