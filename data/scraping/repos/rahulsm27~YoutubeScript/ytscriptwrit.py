import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import os

os.environ['OPENAI_API_KEY'] = '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'

#APP Framework


st.title("Youtube Script Generate with Langcahin")
title = st.text_input("Kindly enter title here")

llm = OpenAI(temperature = 0.9)

#Show stuff on stemalit app
prompt = PromptTemplate.from_template(""" Write a youtube script for the following title \n{title} """) 

chain = LLMChain(llm=llm, prompt=prompt)

if title:

    response = chain.run(title=title)
    st.write(response)



else:
    pass
