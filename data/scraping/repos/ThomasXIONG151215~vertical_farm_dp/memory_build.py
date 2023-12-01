from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationEntityMemory
from langchain.memory.prompt import ENTITY_MEMORY_CONVERSATION_TEMPLATE
from langchain.chat_models import ChatOpenAI
from pydantic import BaseModel
from typing import List, Dict, Any

api = "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"
serapi = "e49a7c10888561af955ab5fcf880d3612e7484e28454f43ad8cf5244e48aeda4"

def entity_memory_conversation():
    #setting up memory
    llm = ChatOpenAI(temperature=0,model="gpt-3.5-turbo-0613",openai_api_key=api)
    memory = ConversationEntityMemory(llm=llm, return_messages=True)
    
    #set birth memory and identity
    #嵌入memory的记忆不再做数
    memory = ConversationEntityMemory(llm=llm)
    _input = {"input": "Deven & Sam are working on a hackathon project"}
    memory.load_memory_variables(_input)
    memory.save_context(
        _input,
        {"output": " That sounds like a great project! What kind of project are they working on?"}
    )

    #调用对话-对象历史
    #memory.load_memory_variables({"input": "What is your name"})

    #对话链应用
    

    conversation = ConversationChain(
        llm=llm, 
        verbose=True,
        prompt=ENTITY_MEMORY_CONVERSATION_TEMPLATE,
        memory=ConversationEntityMemory(llm=llm),
    )
    #对话形式交流
    #conversation.predict(input="What is your name")

    #查看记忆信息
    from pprint import pprint
    #pprint(conversation.memory.entity_store.store)
    #每次新做predict，entity_store就会更新
    
    
    return conversation



