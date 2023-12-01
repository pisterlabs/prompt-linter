import os
import openai
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory


openai.api_key = os.environ['OPENAI_API_KEY']

from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=1)
llm = ChatOpenAI(temperature=0.0)

conversation = ConversationChain(
    llm=llm,
    memory = memory,
    verbose=False
)

memory.save_context({"input": "Hi"},
                    {"output": "What's up"})
memory.save_context({"input": "Not much, just hanging"},
                    {"output": "Cool"})
memory.save_context({"input": "Not much, just hanging1"},
                    {"output": "Cool1"})

# print (memory.load_memory_variables({}))


print (conversation.predict(input="Hi, my name is Andrew"))

print (conversation.predict(input="What is 1+1?"))

print (conversation.predict(input="What is my name?"))