from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.chat_models.openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

chat = ChatOpenAI()
memory = ConversationBufferWindowMemory(k=1)
chain = ConversationChain(llm=chat, memory=memory)

response1 = chain.invoke(input="Hey there! I'm Adam")
print(f"AI: {response1['response']}")  # Hi Adam!

response2 = chain.invoke(input="Hold on.")
print(f"AI: {response2['response']}")  # Likewise, how can I help you?

response3 = chain.invoke(input="Do you know my name?")
print(f"AI: {response3['response']}")  # Nope.
