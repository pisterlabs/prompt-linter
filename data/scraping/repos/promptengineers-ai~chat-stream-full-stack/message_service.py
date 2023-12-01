"""Message service for the chatbot."""
import asyncio
import json
import os
import traceback
from typing import AsyncIterable

from langchain.agents import load_tools
from langchain.callbacks import AsyncIteratorCallbackHandler
from langchain.callbacks.streaming_stdout_final_only import \
    FinalStreamingStdOutCallbackHandler

from src.config import GOOGLE_API_KEY, GOOGLE_CSE_ID
from src.services.chain_service import ChainService
from src.services.loader_service import load_vectorstore
from src.services.logging_service import logger
from src.services.model_service import (chat_model,
                                        openai_chat_functions_model,
                                        openai_chat_model)
from src.services.storage_service import StorageService
from src.utils import wrap_done

os.environ['GOOGLE_CSE_ID'] = GOOGLE_CSE_ID
os.environ['GOOGLE_API_KEY'] = GOOGLE_API_KEY

def token_stream(token: str):
    """ Use server-sent-events to stream the response"""
    data = {
		'sender': 'assistant',
		'message': token,
		'type': 'stream'
	}
    logger.debug('[POST /chat] Stream: %s', str(data))
    return f"data: {json.dumps(data)}\n\n"


def end_stream():
    """Send the end of the stream"""
    end_content = {
    	'sender': 'assistant',
    	'message': "",
    	'type': 'end'
    }
    logger.debug('[POST /chat] End: %s', str(end_content))
    return f"data: {json.dumps(end_content)}\n\n"

def retrieve_system_message(messages):
    """Retrieve the system message"""
    try:
        return list(
			filter(lambda message: message['role'] == 'system', messages)
    	)[0]['content']
    except IndexError:
        return None

def retrieve_chat_messages(messages):
    """Retrieve the chat messages"""
    return [
        (msg["content"]) for msg in messages if msg["role"] in ["user", "assistant"]
    ]
    
#######################################################
## Langchain Chat GPT
#######################################################
async def send_message(
    messages,
    model:str,
    temperature: float or int = 0.9,
) -> AsyncIterable[str]:
    """Send a message to the chatbot and yield the response."""
    callback = AsyncIteratorCallbackHandler()
    model = chat_model(
        model_name=model,
        temperature=temperature,
        streaming=True,
        callbacks=[callback],
    )
    # Begin a task that runs in the background.
    task = asyncio.create_task(wrap_done(
        model.apredict_messages(messages=messages),
        callback.done),
    )

    async for token in callback.aiter():
        # Use server-sent-events to stream the response
        yield token_stream(token)
    yield end_stream()
    await task

#######################################################
## Open AI Chat GPT
#######################################################
async def send_openai_message(
    messages,
    model:str,
    temperature: float or int = 0.9,
) -> AsyncIterable[str]:
    """Send a message to the chatbot and yield the response."""
    response = openai_chat_model(
        messages=messages,
        model_name=model,
        temperature=temperature,
        streaming=True,
    )
    print(response)
    for chunk in response:
        token = chunk['choices'][0]['delta'].get('content', '')
        yield token_stream(token)
    yield end_stream()

#######################################################
## Chat GPT
#######################################################
async def send_functions_message(
    messages,
    model:str,
    temperature: float or int = 0.9,
    functions: list[str] = [],
) -> AsyncIterable[str]:
    """Send a message to the chatbot and yield the response."""
    response = openai_chat_functions_model(
        messages=messages,
        model_name=model,
        temperature=temperature,
        streaming=True,
        keys=functions,
    )
    for chunk in response:
        token = chunk['choices'][0]['delta'].get('content', '')
        yield token_stream(token)
    yield end_stream()

#######################################################
## Vectorstore
#######################################################
async def send_vectorstore_message(
    messages,
    vectorstore,
    model: str,
    temperature: float or int = 0.9,
) -> AsyncIterable[str]:
    """Send a message to the chatbot and yield the response."""
    filtered_messages = retrieve_chat_messages(messages)
    # Retrieve the chat history
    chat_history = list(zip(filtered_messages[::2], filtered_messages[1::2]))
    # Retrieve the system message
    system_message = retrieve_system_message(messages)
    # Create the callback
    callback = AsyncIteratorCallbackHandler()
    # Create the model
    model = chat_model(
        model_name=model,
        temperature=temperature,
        callbacks=[callback],
        streaming=True,
    )
    # Create the query
    query = {'question': filtered_messages[-1], 'chat_history': chat_history}
    # Retrieve the conversation
    qa_chain = ChainService(model).conversation_retrieval(vectorstore, system_message)
    # Begin a task that runs in the background.
    task = asyncio.create_task(wrap_done(
        qa_chain.acall(query),
        callback.done),
    )
	# Yield the tokens as they come in.
    async for token in callback.aiter():
        yield token_stream(token)
    yield end_stream()
    await task

#######################################################
## Agent
#######################################################
def send_agent_message(
    messages,
    model:str,
    temperature: float or int = 0.9,
):
    """Send a message to the chatbot and yield the response."""
    # Retrieve the chat messages
    filtered_messages = retrieve_chat_messages(messages)
    # Retrieve the chat history
    chat_history = list(zip(filtered_messages[::2], filtered_messages[1::2]))              
    # Create the model
    model = chat_model(
        model_name=model,
        temperature=temperature,
        callbacks=[FinalStreamingStdOutCallbackHandler()]
    )
    tools = load_tools(["google-search", "llm-math"],llm=model)
    agent_executor = ChainService(model).agent_search(tools, chat_history)
    try:
        response = agent_executor.run(filtered_messages[-1])
    except BaseException as err:
        tracer = traceback.format_exc()
        logger.error('Error: %s\n%s', err, tracer)
        response = str(err)
        if response.startswith("Could not parse LLM output: "):
            response = response.removeprefix("Could not parse LLM output: ")
	# Yield the tokens as they come in.
    for token in response:
        yield token_stream(token)
    yield end_stream()
