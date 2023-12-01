import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 
from langchain.llms import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import load_tools
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)



st.title("STARTUP STORYTELLER APP")

st.markdown('**This app is designed to help you create a personal storyline on your painpoint and a pitch script for your startup. **')

#define company and project

open_api_key = st.text_input('Enter your open api key. This information is not recorded or stored in any way', type = "password")

name = st.text_input('Enter the name of the person pitching')
painpoint = st.text_input('Enter the painpoint that your startup is solving')
personalconnection = st.text_input('Describe your personal connection to the painpoint')
valueproposition = st.text_input('Describe the value proposition of your startup')


clicked = st.button('Click me')

if clicked:
    st.write('Button clicked! Performing an operation...')
    chat = ChatOpenAI(openai_api_key = open_api_key, model_name='gpt-4', temperature = 0.2)
    # Prompt storytelling on painpoint
    template = "You are an expert in painpoint storytelling. When developing the story, make it as personal as possible, considering my name is {name} and my personal situaton is: {personalconnection}. Make sure to stay within maximum token limit. Focus on the painpoint, not the solution."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template="Here is the painpoint {painpoint}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chatoutput = chat(chat_prompt.format_prompt(name = name, personalconnection = personalconnection, painpoint = painpoint).to_messages())
    st.markdown('**Personal storytelling on painpoint =**')
    st.write(chatoutput)
    # Prompt pitching of value proposition
    pitch_template = "Create a script for a two minute pitch for a startup. I will provide you the painpoint and value proposition. Make sure to stay within maximum token limit."
    pitch_system_message_prompt = SystemMessagePromptTemplate.from_template(pitch_template)
    pitch_human_template="Here is the painpoint: {painpoint}. Here is the value proposition: {valueproposition}"
    pitch_human_message_prompt = HumanMessagePromptTemplate.from_template(pitch_human_template)
    pitch_prompt = ChatPromptTemplate.from_messages([pitch_system_message_prompt, pitch_human_message_prompt])
    pitch_output = chat(pitch_prompt.format_prompt(painpoint = painpoint, valueproposition= valueproposition).to_messages())
    st.markdown('**Pitch script =**')
    st.write(pitch_output)
else:
    st.write('Please click the button to perform an operation')
