import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain,SimpleSequentialChain,SequentialChain
from langchain.memory import ConversationBufferMemory

os.environ['OPENAI_API_KEY']= apikey

#App Framework

st.title(' 👉 Youtube GPT Creator 👈 ')
prompt = st.text_input("Provide your prompt here")


#prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'],
    template="write me a youtube video title about {topic}"
)

script_template = PromptTemplate(
    input_variables = ['title'],
    template="write me a youtube video script for the title {title}"
)

#memory

memory= ConversationBufferMemory(input_key='topic',memory_key='chat_history')



#llm
llm = OpenAI(temperature=0.9)

title_chain = LLMChain(llm=llm, prompt=title_template,verbose=True,output_key='title')
script_chain = LLMChain(llm=llm, prompt=script_template,verbose=True,output_key='script')

# sequential_chain=SimpleSequentialChain(chains=[title_chain,script_chain],verbose=True) # for single output

sequential_chain=SequentialChain(chains=[title_chain,script_chain],verbose=True,input_variables=['topic'],output_variables=['title','script'])

#show stuff if theres a prompt
if prompt:
    # response = sequential_chain.run(prompt) #for single output
    # st.write(response) # for single output
    response = sequential_chain({'topic':prompt})
    st.write(response['title'])
    st.write(response['script'])

    with st.expander('Message History'):
        st.info(memory.buffer)