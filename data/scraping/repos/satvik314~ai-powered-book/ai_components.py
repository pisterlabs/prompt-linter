from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

import os
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model_name = 'gpt-3.5-turbo', temperature= 0.0)
memory = ConversationBufferMemory()

llm_agent = ConversationChain(
    llm = llm,
    memory= memory,
)



