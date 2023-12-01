#integrate code with open api
import os
from constants import openai_key
from langchain.llms import OpenAI
import streamlit as st
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory

os.environ['OPENAI_API_KEY'] = openai_key
# initializing streamlit framework

st.title('Scientist Search Results')

input_text = st.text_input('Search Scientist', 'Enter the name of the Scientist here')

#prompt template
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template = "Tell me about the scientist named {name}."
)

#Memory
person_memory = ConversationBufferMemory(input_key='person', output_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='dob', output_key='chat_history')
description_memory = ConversationBufferMemory(input_key='description', output_key='description_history')   


#Openai llms
llms = OpenAI(temperature=0.8)
#chain
chain = LLMChain(llm=llms, prompt=first_input_prompt, 
                 verbose=True, output_key='person',
                 memory=person_memory)

second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template = "When was {person} born"
)

chain2 = LLMChain(llm=llms, prompt=second_input_prompt, 
                  output_key = 'dob', memory=dob_memory)


third_input_prompt = PromptTemplate(
    input_variables=['dob'],
    template = "Mention 5 major events around {dob} in the world"

) 

chain3 = LLMChain(llm=llms, prompt=third_input_prompt,
                   verbose=True, output_key='description',
                   memory=description_memory)


parent_chain = SequentialChain(chains=[chain,chain2,chain3], 
                               input_variables=['name'],
                               output_variables=['person','dob', 'description'],
                               verbose=True)


if input_text:
    st.write(parent_chain({'name':input_text}))
    
    with st.expander('Person Name'):
        st.info(person_memory.buffer)

    with st.expander('Major Events'):
        st.info(description_memory.buffer)