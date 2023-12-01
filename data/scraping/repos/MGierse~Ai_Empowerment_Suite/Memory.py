from langchain.memory import ConversationBufferMemory

def getConversationBufferMemory(memory_key):
    memory = ConversationBufferMemory(memory_key=memory_key)
    return memory