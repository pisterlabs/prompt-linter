import asyncio
import websockets
import json
import openai
import base64
import shutil
import os
import subprocess
import time
from threading import Thread
from queue import Queue, Empty
from collections.abc import Generator
from langchain.callbacks.base import BaseCallbackHandler
from langchain.llms import OpenAI, OpenAIChat
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI

# Define API keys and voice ID
#OPENAI_API_KEY = '<OPENAI_API_KEY>'
ELEVENLABS_API_KEY = 'key'
VOICE_ID = '21m00Tcm4TlvDq8ikWAM'

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = 'key'


def is_installed(lib_name):
    return shutil.which(lib_name) is not None

async def text_chunker(chunks):
    """Split text into chunks, ensuring to not break sentences."""
    splitters = (".", ",", "?", "!", ";", ":", "—", "-", "(", ")", "[", "]", "}", " ")
    buffer = ""
    
    async for text in chunks:
        if buffer.endswith(splitters):
            yield buffer + " "
            buffer = text
        elif text.startswith(splitters):
            yield buffer + text[0] + " "
            buffer = text[1:]
        else:
            buffer += text
    if buffer:
        yield buffer + " "

async def stream(audio_stream):
    """Stream audio data using mpv player."""
    if not is_installed("mpv"):
        raise ValueError(
            "mpv not found, necessary to stream audio. "
            "Install instructions: https://mpv.io/installation/"
        )
        
    mpv_process = subprocess.Popen(
        ["mpv", "--no-cache", "--no-terminal", "--", "fd://0"],
        stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    
    print("Started streaming audio")
    async for chunk in audio_stream:
        if chunk:
            mpv_process.stdin.write(chunk)
            mpv_process.stdin.flush()
            
    if mpv_process.stdin:
        mpv_process.stdin.close()
    mpv_process.wait()

async def text_to_speech_input_streaming(voice_id, text_iterator):
    """Send text to ElevenLabs API and stream the returned audio."""
    uri = f"wss://api.elevenlabs.io/v1/text-to-speech/{voice_id}/stream-input?model_id=eleven_monolingual_v1"
    
    async with websockets.connect(uri) as websocket:
        await websocket.send(json.dumps({
            "text": " ",
            "voice_settings": {"stability": 0.5, "similarity_boost": True},
            "xi_api_key": ELEVENLABS_API_KEY,
        }))
        
        async def listen():
            """Listen to the websocket for audio data and stream it."""
            while True:
                try:
                    message = await websocket.recv()
                    data = json.loads(message)
                    if data.get("audio"):
                        yield base64.b64decode(data["audio"])
                    elif data.get('isFinal'):
                        break
                except websockets.exceptions.ConnectionClosed:
                    print("Connection closed")
                    break
                
        listen_task = asyncio.create_task(stream(listen()))
        
        async for text in text_chunker(text_iterator):
            await websocket.send(json.dumps({"text": text, "try_trigger_generation": True}))
        
        await websocket.send(json.dumps({"text": ""}))
        
        await listen_task

class QueueCallback(BaseCallbackHandler):
    def __init__(self, q):
        self.q = q
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.q.put(token)
    def on_llm_end(self, *args, **kwargs) -> None:
        return self.q.empty()

def langchain_stream(input_text) -> Generator:

    # Create a Queue
    q = Queue()
    job_done = object()

    # Initialize the LLM we'll be using
    prompt ='''you are an receptionist at taj hotel mumbai.
                user question : {input}
                response:
                    '''
    template = PromptTemplate(template=prompt,input_variables=['input'])
    
    llm = ChatOpenAI(
        model ='gpt-3.5-turbo',
        streaming=True, 
        callbacks=[QueueCallback(q)], 
        temperature=0.1
    )
    chain = LLMChain(llm=llm,prompt=template)
    # Create a funciton to call - this will run in a thread
    def task():
        resp = chain.run(input_text)
        q.put(job_done)

    # Create a thread and start the function
    t = Thread(target=task)
    t.start()

    content = ""

    # Get each new token from the queue and yield for our generator
    while True:
        try:
            next_token = q.get(True, timeout=1)
            if next_token is job_done:
                break
            content += next_token
            yield next_token
        except Empty:
            continue

async def chat_completion(chat_history):
    user_input = chat_history[-1]['content']
    final_text = ""
    async def text_iterator():
        nonlocal final_text
        #print(type(langchain_stream(user_input)))
        for token in langchain_stream(user_input):
            print(token)
            final_text += token
            yield token
    await text_to_speech_input_streaming(VOICE_ID, text_iterator())
    return final_text

# Main execution
if __name__ == "__main__":
    user_query  = [{'role': 'system', 'content': """
                                        'user': tell me about the history of this place
                                        """}]
    s = time.time()
    final_response = asyncio.run(chat_completion(user_query))
    t = time.time()
    print(f"Time taken: {t-s}")
    print(f"Final text from GPT: {final_response}")
