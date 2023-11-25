# Description: A Streamlit app that uses the OpenAI API to generate a reponse that talks like a pirate.

# Import the required libraries
import os 
from apikey import apikey 

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory

# Set the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey

# Set the title of the app
st.title('🦜🔗 Mark\'s Pirate ChatGPT')

# Inject custom CSS to adjust font size of text_input
st.markdown(
    """
    <style>
        .stTextInput > div > div > input {
            font-size: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# Set the prompt for the user to enter
prompt = st.text_input('Your Prompt Textbox Me Hearties:', placeholder='Enter ye words here, Savvy?') 

# Define prompt template for generating responses like a pirate
subject_template = PromptTemplate(
    input_variables = ['subject'], 
    template='talk like a pirate about {subject}'
)

# Define prompt template for generating discussions like a pirate
script_template = PromptTemplate(
    input_variables = ['title'], 
    template='discuss like a pirate {title}'
)

# Set up memory buffers for storing previous conversations
subject_memory = ConversationBufferMemory(input_key='subject', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


# Set up the OpenAI language model and LLM chains for generating titles and scripts
llm = OpenAI(temperature=0.9) 
subject_chain = LLMChain(llm=llm, prompt=subject_template, verbose=True, output_key='title', memory=subject_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

# Show the prompt and generated content on the screen if there is a prompt
if prompt: 
    title = subject_chain.run(prompt)
    script = script_chain.run(title=title)

    # Show the title and script on the screen
    st.write(title) 
    st.write(script) 
    
    # Show the subject and script history in an expander
    with st.expander('Subject History'): 
        st.info(subject_memory.buffer)
    with st.expander('Script History'): 
        st.info(script_memory.buffer)