'''
Description: 
Author: colin gao
Date: 2023-05-14 17:16:12
LastEditTime: 2023-05-26 23:06:49
'''
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.prompts.prompt import PromptTemplate

from configs.config import *

from dotenv import load_dotenv
load_dotenv()

import code

def get_chain():
    conversation_memory = ConversationBufferWindowMemory(
        memory_key="history",
        k=5
    )

    chat_llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=TEMPERTURE
    )

    DEFAULT_TEMPLATE = """这是一个专门用于回答占卜相关问题的工具。只要你提出与占卜相关的问题，或者明确说出"占卜"，这个工具就会被启动来寻找最合适的答案。无论是初次的占卜询问，还是后续的深入探讨，这个工具都可以提供协助。
    最重要的一点，这个工具占卜的方式是周易占卜，针对所有的问题，都是通过聊天的模式实现周易占卜。

    Current conversation:
    {history}
    Human: {input}
    AI:"""
    PROMPT = PromptTemplate(input_variables=["history", "input"], template=DEFAULT_TEMPLATE)

    conversation = ConversationChain(
        llm=chat_llm,
        memory=conversation_memory,
        prompt=PROMPT
    )

    return conversation


if __name__ == "__main__":
    from colorama import init, Fore, Style
    init()

    conversation_chat = get_chain()

    while True:
        question = input("Please enter your question (or type 'exit' to end): ")
        if question.lower() == 'exit':
            break
        
        result = conversation_chat(question)

        print(f'{Fore.BLUE}{Style.BRIGHT}AI:{Fore.RESET}{Style.NORMAL} {result["response"]}')
