import os
from openai import AzureOpenAI
import asyncio

AZURE_OPENAI_VERSION = os.environ.get("AZURE_OPENAI_VERSION", None)
AZURE_OPENAI_DEPLOYMENT = os.environ.get("AZURE_OPENAI_DEPLOYMENT", None)
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")

def azure_openai_llm_handler(messages: list, stream: bool = False):
    client = AzureOpenAI(
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_key=AZURE_OPENAI_KEY,
        api_version=AZURE_OPENAI_VERSION,
    )

    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT, 
        messages=messages,
        stream=stream  # This will be True or False based on the argument
    )

    if stream:
        # Convert the response to an async iterable if streaming
        return async_wrapper(response)
    else:
        # Return the response directly if not streaming
        return response

async def async_wrapper(sync_iterable):
    for item in sync_iterable:
        yield item
        await asyncio.sleep(0)  # Yield to the event loop

