from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv

load_dotenv()
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)

from langchain.chains import LLMChain

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a chatbot speaking in pirate english."
        ),  # The persistent system prompt
        MessagesPlaceholder(
            variable_name="chat_history"
        ),  # Where the memory will be stored.
        HumanMessagePromptTemplate.from_template(
            "{human_input}"
        ),  # Where the human input will injected
    ]
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

llm = ChatOpenAI()

chat_llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=True,
    memory=memory,
)

response = chat_llm_chain.predict(human_input="Hi there my friend")

print(response)

response = chat_llm_chain.predict(human_input="How are you today?`")

print(response)
