import os
from langchain.llms import OpenAI
#from langchain import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
import streamlit as st

st.title("Youtube Title Generator App!!")
prompt = st.text_input("Write your keyword here")

#print(prompt)
open_ai_key = input("Enter your api key:")
llm = OpenAI(temperature=0.9, openai_api_key=open_ai_key)

title_template = PromptTemplate(
    input_variables = ["keyword"],
    template = "write a youtube video title about {keyword}")


desc_template = PromptTemplate(
    input_variables = ["title"],
    template = "based on the {title} of the youtube video, generate a youtube description in atmost 150 words"
)

title_chain = LLMChain(llm=llm, prompt=title_template, output_key="title")
desc_chain = LLMChain(llm=llm, prompt=desc_template, output_key="description")
#sequential_chain = SimpleSequentialChain(chains=[title_chain, desc_chain], verbose=True)

sequential_chain = SequentialChain(chains=[title_chain, desc_chain], input_variables=["keyword"], 
                                  output_variables=["title","description"], verbose=True)

if prompt:
    response = sequential_chain({"keyword":prompt})
    st.write(response["title"])
    st.write(response["description"])
    