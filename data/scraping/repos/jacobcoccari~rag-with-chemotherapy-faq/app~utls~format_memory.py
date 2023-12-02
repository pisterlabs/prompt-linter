from langchain.memory import ChatMessageHistory

def get_chat_history(streamlit_memory):
    str = ""
    for message in streamlit_memory:
        str += message["role"] + ": " + message["content"] + "\n"
    return str



        
