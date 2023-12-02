from langchain.memory import ConversationBufferMemory

def get_memory_short():
    from langchain.memory.chat_message_histories.in_memory import ChatMessageHistory
    message_history = ChatMessageHistory()
    memory = ConversationBufferMemory(memory_key="history", chat_memory=message_history, output_key="output", input_key="input", return_messages=True)
    return memory