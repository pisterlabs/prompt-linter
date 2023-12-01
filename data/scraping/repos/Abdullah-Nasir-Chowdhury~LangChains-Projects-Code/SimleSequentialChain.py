import os
import openai
from dotenv import main

main.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

from langchain.llms import OpenAI
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain

st.title('YouTube GPT Generator')
prompt = st.text_input('Plug in your prompt here!')

title_template = PromptTemplate(input_variables=['topic'], 
                                template='Write me a youtube video title about {topic}')
script_template = PromptTemplate(input_variables=['title'],
                                 template='Write me a script based on the video title: {title}')


llm = OpenAI(temperature=0)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True)
sequential_chain = SimpleSequentialChain(chains=[title_chain, script_chain], verbose=True)

if st.button(label='Generate'):
    if prompt:
        response = sequential_chain.run(prompt)
        st.write(response)
    else:
        st.write('Please enter a prompt!')