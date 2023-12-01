import asyncio
from fastapi import FastAPI, WebSocket
from concurrent.futures import ThreadPoolExecutor
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
import os

os.environ["OPENAI_API_KEY"] = "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"

app = FastAPI()

llm = ChatOpenAI(temperature=0.0)
math_llm = OpenAI(temperature=0.0)

# Test case:
# Iput: "What's my friend Eric's surname?"
# model 

# agent with tool
tools = load_tools(["ddg-search", "human"], llm=math_llm) 
agent_chain = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION | AgentType.HUMAN,
    verbose=True,
    handle_parsing_errors=True,
)

executor = ThreadPoolExecutor() # to make websockets work properly, use Thread to run each request in different threads
lock = asyncio.Lock()  # Create a lock

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        req = await websocket.receive_text()
        isErrorOccured = True # this is for error handling (even error occurs, connection doesn't close)
        response = ""
        while isErrorOccured:
            try:
                async with lock:  # Acquire the lock
                    response = await asyncio.get_event_loop().run_in_executor(
                        executor, agent_chain.run, req
                    )
                    isErrorOccured = False # I got response without error -> go out of while statement and send response
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                isErrorOccured = True # there's an error, will retry to get response
        await websocket.send_text(response)