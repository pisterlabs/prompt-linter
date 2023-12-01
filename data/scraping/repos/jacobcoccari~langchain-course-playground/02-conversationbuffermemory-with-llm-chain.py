from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import load_dotenv

load_dotenv()

from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(return_messages=True)

llm = ChatOpenAI()
conversation = LLMChain(
    llm=llm,
    verbose=True,
    memory=ConversationBufferMemory(),
)

result = conversation.predict(input="Hi there!")

print(result)


result = conversation.predict(input="My name is jacob")

print(result)

result = conversation.predict(input="what is my name?")

print(result)
