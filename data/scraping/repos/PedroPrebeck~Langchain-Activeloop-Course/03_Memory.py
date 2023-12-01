from dotenv import load_dotenv

load_dotenv()

from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

llm = OpenAI(model="text-davinci-003", temperature=0)

conversation = ConversationChain(
    llm=llm, verbose=True, memory=ConversationBufferMemory()
)

conversation.predict(input="Tell me about yourself.")

conversation.predict(input="What can you do?")

conversation.predict(input="How can you help me with data analysis?")

print(conversation)
