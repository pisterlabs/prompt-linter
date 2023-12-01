import os
from dotenv import load_dotenv
import openai
from langchain import PromptTemplate
from langchain.llms import OpenAI
from langchain import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.memory import ConversationBufferMemory
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 

# Streamlit framework
st.title('Langchain Application')

def lang_tut():
    input_text = st.text_input("Search the topic you want")
    # OpenAI LLMs 
    llm = OpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.9) 

    # Memory

    person_memory = ConversationBufferMemory(input_key='name', memory_key='chat_history')
    dob_memory = ConversationBufferMemory(input_key='person', memory_key='chat_history')
    descr_memory = ConversationBufferMemory(input_key='dob', memory_key='description_history')


    # Prompt one Template
    prompt_one = PromptTemplate(input_variables=['name'],
                                template="give me a brief history of {name}") 
    # Initialize LLMchain 1
    chain_1 = LLMChain(llm=llm, prompt=prompt_one, verbose=True, output_key='About',memory=person_memory) 

    # Prompt two Template    
    prompt_two = PromptTemplate(input_variables=['About'], 
                                template="when and where {About} was born?")   
    # Initialize LLMchain 2
    chain_2 = LLMChain(llm=llm, prompt=prompt_two, verbose=True, output_key='dob', memory=dob_memory)

    # Prompt three Template    
    prompt_three = PromptTemplate(input_variables=['About'],   
                                template="Who are similar personalities to {About} ")   
    # Initialize LLMchain 3
    chain_3 = LLMChain(llm=llm, prompt=prompt_three, verbose=True, output_key='similar_person',memory=descr_memory)

    # Create a chain
    parent_chain = SimpleSequentialChain(chains=[chain_1, chain_2, chain_3], input_key='name', output_key= 'similar_person', verbose=True)

    if input_text:
        st.write(parent_chain.run({'name': input_text}))

if __name__ == "__main__":
    lang_tut()
