import os
import time

from util import config_util

config = config_util.ConfigClsf().get_config()
openai_api_key = os.getenv('OPENAI_API_KEY', config['OPENAI']['API'])

from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationSummaryMemory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI



def conversation_buffer():
    # 메모리 생성
    memory = ConversationBufferMemory()
    memory.chat_memory.add_user_message("배고프다")
    memory.chat_memory.add_ai_message("어디 가서 밥 먹을까?")
    memory.chat_memory.add_user_message("라면 먹으러 가자")
    memory.chat_memory.add_ai_message("지하철역 앞에 있는 분식집으로 가자")
    memory.chat_memory.add_user_message("그럼 출발!")
    memory.chat_memory.add_ai_message("OK!")

    # 변수 가져오기
    load_memory = memory.load_memory_variables({})
    print(load_memory['history'])



#최근 K개의 대화를 가져오는 메모리
def conversation_window():

    memory = ConversationBufferWindowMemory(k=2, return_messages=True)
    memory.save_context({"input": "안녕!"}, {"ouput": "무슨 일이야?"}) # 최근 2개만, 안나옴
    memory.save_context({"input": "배고파"}, {"ouput": "나도"})
    memory.save_context({"input": "밥 먹자"}, {"ouput": "OK!"})

    load_memory = memory.load_memory_variables({})
    print(load_memory['history'])


# 대화 기록을 요약해서 저장하는 메모리
def conversation_summary():

    memory = ConversationSummaryMemory(llm=ChatOpenAI(temperature=0,openai_api_key=openai_api_key), return_messages=True, )
    memory.save_context({"input": "배고파"}, {"ouput": "어디 가서 밥 먹을까?"})
    memory.save_context({"input": "라면 먹으러 가자"}, {"ouput": "역 앞에 있는 분식집으로 가자"})
    memory.save_context({"input": "그럼 출발!"}, {"ouput": "OK!"})

    load_memory = memory.load_memory_variables({})
    print(load_memory['history'])


# 대화 기록의 요약과 최근 대화 K개를 같이 활용하는 메모리
def conversation_summary_buffer():
    # 메모리 생성
    memory = ConversationSummaryBufferMemory(
        llm=ChatOpenAI(temperature=0, openai_api_key=openai_api_key),
        max_token_limit=50,
        return_messages=True
    )
    memory.save_context({"input": "안녕"}, {"ouput": "무슨 일이야?"})
    memory.save_context({"input": "배고파"}, {"ouput": "나도"})
    memory.save_context({"input": "밥 먹자"}, {"ouput": "OK!"})

    # 변수 가져오기
    load_memory = memory.load_memory_variables({})
    print(load_memory['history'])


# 메모리 기반 대화 생성
def conversation_chain():

    # 메모리 생성
    memory = ConversationBufferWindowMemory(k=2, return_messages=True)
    memory.save_context({"input": "안녕"}, {"ouput": "무슨 일이야?"})
    memory.save_context({"input": "배고파"}, {"ouput": "그래?"})

    # 변수 가져오기
    load_memory = memory.load_memory_variables({})
    print(load_memory['history'])

    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    conversation = ConversationChain(
        llm=llm,
        verbose=True,
        memory=memory
    )

    print(conversation.run(input="대화를 내용에 대해 이야기하세요"))

    print(conversation.run(input="다시한번 말해줘요"))

    print(conversation.run(input="배고프니?"))


if __name__=="__main__":
    # conversation_buffer()
    conversation_chain()
    pass