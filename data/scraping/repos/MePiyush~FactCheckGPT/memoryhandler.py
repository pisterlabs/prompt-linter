from langchain.memory import ConversationBufferMemory

class memoryhandler:
    def __init__(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history",output_key='answer',return_messages=True)

    def get_memory(self):
        return self.memory