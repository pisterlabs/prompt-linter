from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import (
    HumanMessage,
    AIMessage,
    SystemMessage
)
import time
import os
OPENAI_API_KEY,API_BASE = os.getenv('OPENAI_API_KEY'),os.getenv('API_BASE')
from langchain.chat_models import ChatOpenAI
from jinja2 import Template


gpt_35 = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()],temperature = 0.0 , openai_api_key = OPENAI_API_KEY, openai_api_base = API_BASE, 
                                              model_name='gpt-3.5-turbo-16k-0613' )
gpt4 = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()],temperature = 0.0 , openai_api_key = OPENAI_API_KEY, openai_api_base = API_BASE, 
                                              model_name='gpt-4-0314' )


def new_gpt35(temperature=0.0,**kwargs):
    if temperature == 0.0 and len(kwargs) == 0:
        return gpt_35
    llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()],temperature = temperature , openai_api_key = OPENAI_API_KEY, openai_api_base = API_BASE, 
                                              model_name='gpt-3.5-turbo-16k-0613' ,**kwargs)
    return llm

def new_gpt4(temperature=0.0,**kwargs):
    if temperature == 0.0 and len(kwargs) == 0:
        return gpt4
    llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()],temperature = temperature , openai_api_key = OPENAI_API_KEY, openai_api_base = API_BASE, 
                                              model_name='gpt-4-0314' ,**kwargs)
    return llm  





def get_token_cnt(text):
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    enctext =enc.encode(text)
    return len(enctext)


def jinja_format(message, **kwargs):
    return Template(message).render(**kwargs)


class BaseLLM:
    def __init__(self, llm) -> None:
        
        self.llm = llm
        self.max_retries = 5  # 设置最大重试次数
        self.retry_delay = 1  # 设置每次重试之间的延迟（秒）
        self.msgs = []

    def add_human(self,msg):
        self.msgs.append(HumanMessage(content=msg))

    def add_system(self,msg):
        self.msgs.append(SystemMessage(content=msg))
    
 
    def clear(self):
        self.msgs = []
    
    def add_AI(self,resp):
        self.msgs.append(AIMessage(content=resp))

    def get_msgs(self):
        return self.msgs

    def set_msgs(self,msgs):
        self.msgs = msgs

    


class LLM(BaseLLM):
    def __init__(self, llm) -> None:
        super().__init__(llm)
    
    def get_reply(self):

        AImsg = self.llm(self.msgs)
        resp = AImsg.content
        return resp

    def get_time_cost_reply(self):
        
        start = time.time()
        resp = self.get_reply()
        end = time.time()
        print(f'cost time: {end-start}')
        
        return resp

