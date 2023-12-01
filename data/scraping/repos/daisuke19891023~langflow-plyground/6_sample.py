import streamlit as st
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from util.base_ui import ChatUI


# 描画に失敗するため、原則使わない
@st.cache_resource
def load_chain() -> ConversationChain:
    """Logic for loading the chain you want to use should go here."""
    # template = "{history} let's think step by step"
    # prompt = PromptTemplate(input_variables=["history"], template=template)
    chat = ChatOpenAI()
    # chain = LLMChain(llm=chat, prompt=load_translate_prompt(), verbose=True)
    chain = ConversationChain(llm=chat, memory=ConversationBufferMemory(), verbose=True)
    return chain


ui = ChatUI(chain=load_chain(), title="sample bot")
ui()
