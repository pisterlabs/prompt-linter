from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_experimental.smart_llm import SmartLLMChain
import os
from dotenv import load_dotenv
import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

hard_question = "can you write me an unsupervised learning algorithm for a neural network that can write code?"
prompt = PromptTemplate.from_template(hard_question)
llm = ChatOpenAI(temperature=0, model_name="gpt-4")
chain = SmartLLMChain(llm=llm, prompt=prompt, n_ideas=3, verbose=True)



if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    
    hard_question = prompt  # Directly use the prompt as a string
    
with st.chat_message("assistant"):
    st_callback = StreamlitCallbackHandler(st.container())
    chain.run({})

    
    
 




