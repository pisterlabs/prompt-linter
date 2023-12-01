from dotenv import load_dotenv

from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

load_dotenv()

llm = ChatOpenAI()

conversation = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory(), 
    verbose=True)

# print(conversation.prompt.template)

print("Hello! How can I help you?")

'''
## Second part  

while True:
    user_input = input("> ")
    ai = conversation(user_input)
    print("AI: ", ai['response'])

'''