from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import LLMChain
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from dotenv import load_dotenv

load_dotenv()

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

memory = ConversationBufferWindowMemory(
    memory_key="chat_history", k=1, return_messages=True
)

llm = ChatOpenAI()

chat_llm_chain = LLMChain(
    llm=llm,
    prompt=prompt,
    memory=memory,
)

response = chat_llm_chain.predict(human_input="My name is jacob")

print(response)

response = chat_llm_chain.predict(human_input="How are you today?`")

response = chat_llm_chain.predict(human_input="What is my name?")

print(response)
