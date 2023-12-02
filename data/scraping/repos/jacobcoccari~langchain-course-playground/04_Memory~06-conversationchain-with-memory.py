from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate,
)

from langchain.chains import ConversationChain
from dotenv import load_dotenv

load_dotenv()

prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a chatbot having a conversation with a human."
        ),  # The persistent system prompt
        MessagesPlaceholder(
            variable_name="history"
        ),  # Where the memory will be stored.
        HumanMessagePromptTemplate.from_template(
            "{input}"
        ),  # Where the user input will be stored.
    ]
)

memory = ConversationBufferMemory(
    return_messages=True,
    # memory_key="history",
)

llm = ChatOpenAI()

chat_llm_chain = ConversationChain(
    llm=llm,
    verbose=True,
    memory=memory,
    # prompt=prompt,
)

chat_llm_chain.predict(input="Hi there my friend")
