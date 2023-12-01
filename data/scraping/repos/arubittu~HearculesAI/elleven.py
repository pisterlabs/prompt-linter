from elevenlabs import generate, stream, set_api_key
from langchain.callbacks.base import BaseCallbackHandler
from langchain.llms import OpenAI, OpenAIChat
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
import base64
import shutil
import os
import subprocess
import time
from threading import Thread
from queue import Queue, Empty
from collections.abc import Generator


set_api_key('key')
os.environ["OPENAI_API_KEY"]  = 'key'

def text_stream():
    yield " "
    yield "okay"
    yield " "
    

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
        model ='gpt-4',
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
            print(next_token)
            yield next_token
        except Empty:
            continue
        
audio_stream = generate(
    text=langchain_stream('tell me more about this place'),
    voice="Nicole",
    model="eleven_monolingual_v1",
    stream=True,
    latency=1,
    stream_chunk_size=128   
)
if __name__=="__main__":
    stream(audio_stream)
    