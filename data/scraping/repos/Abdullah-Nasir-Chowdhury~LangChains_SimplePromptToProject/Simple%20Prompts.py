from typing_extensions import Protocol
import os, openai
from dotenv import main, load_dotenv, find_dotenv
from apikey import api_key

main.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
# print(openai.api_key)
# os.environ['OPENAI_API_KEY'] = api_key

from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, ConversationChain, SimpleSequentialChain, SequentialChain
# from langchain.utilities import 

import streamlit as st

st.title('YouTube GPT Generator')
prompt = st.text_input('Plug in your prompt here!')

llm = OpenAI(temperature=0.0)


if st.button(label='Generate'):   
    if prompt:
        response = llm(prompt)
        st.write(response)
    else:
        st.write('Please enter a prompt!')