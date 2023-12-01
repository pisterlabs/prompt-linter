import os
import openai
from dotenv import main

main.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

from langchain.llms import OpenAI
import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain, SimpleSequentialChain
from langchain.memory import ConversationBufferMemory

st.title('YouTube GPT Generator')
prompt = st.text_input('Plug in your prompt here!')

title_template = PromptTemplate(input_variables=['topic'], 
                                template='Write me a youtube video title about {topic}'
                                )
script_template = PromptTemplate(input_variables=['title'],
                                 template='Write me a script based on the video title: {title}'
                                 )


memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')

llm = OpenAI(temperature=0)

title_chain = LLMChain(llm=llm, 
                       prompt=title_template, 
                       verbose=True, 
                       output_key='title',
                       memory=memory)

script_chain = LLMChain(llm=llm, 
                        prompt=script_template, 
                        verbose=True, 
                        output_key='script',
                        memory=memory)

sequential_chain = SequentialChain(chains=[title_chain, script_chain], 
                                   verbose=True,
                                   input_variables=['topic'],
                                   output_variables=['title','script'])


if st.button(label='Generate'):
    if prompt:
        response = sequential_chain({'topic':prompt})
        st.write(response['title'])
        st.write(response['script'])
        
        with st.expander('Message history'):
            st.info(memory.buffer)
    else:
        st.write('Please enter a prompt!')
        
        