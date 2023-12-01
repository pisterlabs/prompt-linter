import os
from langchain.llms import OpenLLM
from modules.prompt import Prompt
from dotenv import load_dotenv
load_dotenv()
import asyncio

def chat(messages, max_tokens):
    max_tokens = max_tokens
    url = os.environ['API_URL']
    output_msg = Prompt.prepare(messages)
    print(output_msg)
    output_msg += 'Assistant: '

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    llm = OpenLLM(server_url=url)
    response = llm(prompt=output_msg)
    print(response)
    
    loop.close()
    return response
