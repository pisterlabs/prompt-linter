from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain

llm = Ollama(model="dolphin2.2-mistral",
             temperature="0.6",
             system="You are a very helpful AI assistant")

memory = ConversationBufferWindowMemory(k=10)

chain = ConversationChain(llm=llm, memory=memory)

while True:
    print(f"""AI : {chain.run(input("User : "))}""")
