import streamlit as st
from langchain. chat_models import ChatOpenAI
from langchain import PromptTemplate
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import (
    HumanMessage,
)
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import openai
from typing import Any, Dict, List
import os
from dotenv import load_dotenv

# .envファイルの読み込み
load_dotenv(os.path.join(os.path.dirname(__file__), '...', '.env'))

openai_api_key=os.environ["OPENAI_API_KEY"]

st.subheader("WebChatGPT(デモ)&ストリーミング処理実験中")
                                  
def get_state(): 
     if "state" not in st.session_state: 
         st.session_state.state = {"memory": ConversationBufferMemory(memory_key="chat_history")} 
     return st.session_state.state 
state = get_state()

st.write(state['memory'].load_memory_variables({}))

prompt = PromptTemplate(
    input_variables=["chat_history","input"], 
    template='Based on the following chat_history, Please reply to the question in format of markdown. history: {chat_history}. question: {input}'
)

user_input = st.text_input("You: ",placeholder = "Ask me anything ...")
ask = st.button('ask',type='primary')
st.markdown("----")

class SimpleStreamlitCallbackHandler(BaseCallbackHandler):
    """ Copied only streaming part from StreamlitCallbackHandler """
    
    def __init__(self) -> None:
        self.tokens_area = st.empty()
        self.tokens_stream = ""
        
    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Run on new LLM token. Only available when streaming is enabled."""
        self.tokens_stream += token
        self.tokens_area.markdown(self.tokens_stream)

handler = SimpleStreamlitCallbackHandler()

if ask:
    res_box = st.empty()
    with st.spinner('typing...'):
        report = []
        chat = ChatOpenAI(streaming=True, temperature=0.9)
        conversation = ConversationChain(
            llm=chat, 
            prompt=prompt,
            memory=state['memory']            
        )
        res = conversation.predict(input=user_input, callbacks=[handler])
    
st.markdown("----")
