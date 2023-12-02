import os
import time

from util import config_util

config = config_util.ConfigClsf().get_config()
openai_api_key = os.getenv('OPENAI_API_KEY', config['OPENAI']['API'])

from langchain.globals import set_llm_cache
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.cache import InMemoryCache
from langchain.schema import HumanMessage, SystemMessage


# 채팅 기능 # chat message
# System : AI에게 해야 할 일을 알려주는 배경 컨텍스트
# Human : 사용자 메세지
# AI : AI가 응답한 내용을 보여주는 상세 메세지

chat = ChatOpenAI(temperature=.7,
                  max_tokens=200,
                  cache=False,
                  callbacks=([StreamingStdOutCallbackHandler()]),  # 콜백 기능 지원
                  streaming=True,
                  verbose=True,
                  openai_api_key=openai_api_key
                  )


def simple_chat():
    response = chat([
        SystemMessage(content="You are a nice AI bot that helps a user figure out what to eat in one short sentence"),
        HumanMessage(content="I like tomatoes, what should I eat?")
    ])
    print(response)

    # finish_reason 출력하려면...
    response = chat.generate([[
        SystemMessage(content="You are a nice AI bot that helps a user figure out what to eat in one short sentence"),
        HumanMessage(content="I like tomatoes, what should I eat?")
    ]])
    print(response)


def batch_chat():
    # 한번에 여러번 호출하기
    message_list = [HumanMessage(content="고양이 이름 지어줘"),
                    HumanMessage(content="개 이름 지어줘")]
    batch = chat.generate([message_list]) # generate로 한번에 생성한다.
    print(batch)


def function_chat():
    # Function calling 기능
    output = chat(messages=
         [
             SystemMessage(content="You are an helpful AI bot"),
             HumanMessage(content="What’s the weather like in Boston right now?")
         ],

         functions=[{
             "name": "get_current_weather",
             "description": "Get the current weather in a given location",
             "parameters": {
                 "type": "object",
                 "properties": {
                     "location": {
                         "type": "string",
                         "description": "The city and state, e.g. San Francisco, CA"
                     },
                     "unit": {
                         "type": "string",
                         "enum": ["celsius", "fahrenheit"]
                     }
                 },
                 "required": ["location"]
             }
         }
         ]
    )

    print(output)


"""
인메모리 기능은 전역변수에 메모리 캐시를 저장해서 사용하는 기능이 있습니다.
llm 모델에 사용해서 cache를 True / False를 통해 선언해서 필요에 따라 사용할 수 있습니다. 
이미 한번 생성한 질문에 대해서 똑같은 답변을 바로 답변해줍니다.
"""

def memory_cache_chat():
    # 전역변수에 메모리 캐시
    chat = ChatOpenAI(temperature=.7,
                      callbacks=([StreamingStdOutCallbackHandler()]),  # 콜백 기능 지원
                      streaming=True,
                      verbose=True,
                      openai_api_key=openai_api_key
                      )

    set_llm_cache(InMemoryCache())
    start = time.time()
    print(chat.generate([[HumanMessage(content="고양이 이름 지어줘")]]))
    end = time.time()
    print(end-start) # 2초

    start = time.time()
    print(chat.generate([[HumanMessage(content="고양이 이름 지어줘")]]))
    end = time.time()
    print(end-start) # 0.0019991397857666016초


def memory_false_chat():
    chat = ChatOpenAI(temperature=.7,
                      callbacks=([StreamingStdOutCallbackHandler()]),  # 콜백 기능 지원
                      streaming=True,
                      verbose=True,
                      openai_api_key=openai_api_key,
                      cache=False # False로 사용
                      )

    set_llm_cache(InMemoryCache())
    start = time.time()
    print(chat.generate([[HumanMessage(content="고양이 이름 지어줘")]]))
    end = time.time()
    print(end - start)  # 2초

    start = time.time()
    print(chat.generate([[HumanMessage(content="고양이 이름 지어줘")]]))
    end = time.time()
    print(end - start)  # 2초



if __name__=="__main__":
    simple_chat()
    batch_chat()
    function_chat()
    memory_cache_chat()
    memory_false_chat()
