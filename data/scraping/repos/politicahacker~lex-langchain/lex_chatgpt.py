import os

#LLM
from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
#Memory
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

#CallBack
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

#Prompts
from .prompts import SYS_PROMPT
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

#Define o LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content=SYS_PROMPT), # The persistent system prompt
    MessagesPlaceholder(variable_name="chat_history"), # Where the memory will be stored.
    HumanMessagePromptTemplate.from_template("{human_input}"), # Where the human input will injected
])
    
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatOpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()])

chat_llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
)