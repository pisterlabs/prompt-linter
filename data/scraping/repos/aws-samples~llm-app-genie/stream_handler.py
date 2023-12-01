from dataclasses import dataclass
from langchain.callbacks.base import BaseCallbackHandler

# reference https://github.com/streamlit/StreamlitLangChain/blob/main/streaming_demo.py
@dataclass
class StreamHandler(BaseCallbackHandler):
    
    def __init__(self, callback, initial_text=""):
        self.llm_callback = callback
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs):
        self.text += token
        self.llm_callback(self.text)
        