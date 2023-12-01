# Bring in deps
import os 
from apikey import apikey 

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 
import wikipedia
import finnhub

os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('Risk Analysis Generator')
prompt = st.text_input('Plug in your prompt here') 


# Prompt templates
input_template = PromptTemplate(
    input_variables = ['event', 'event_research'], 
    template='Your role is to act as a financial macro-economics researcher. Reply with a detailed estimate of how the event: {event} could \
    affect financial markets or any investment classes, keeping your response below 100 words. \
    For more context on the headline, refer to the following research: {event_research}'
)

script_template = PromptTemplate(
    input_variables = ['input_chain_output', 'portfolio', 'portfolio_research'], 
    template='Your role is to act as a risk analyst. You are given the following \
    situation: {input_chain_output}. Write a detailed risk analysis \
    on how this situation affects the risk rating of the stock portfolio {portfolio}. \
    Do not give advice on how to do risk analysis, instead give an evaluation of the risks posed by the event. \
    Keep your response as technical as possible, and below 200 words. \
    Refer to research on the portfolio {portfolio_research} for more context.'
)


# Memory 
title_memory = ConversationBufferMemory(input_key='raw_input', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='wikipedia_research', memory_key='chat_history')

# Llms
llm = OpenAI(temperature=0.8, presence_penalty=0.2, frequency_penalty= 0.2, max_tokens=1000) 
input_chain = LLMChain(llm=llm, prompt=input_template, verbose=True, output_key='macro_research')
output_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='total_risk_analysis')

wiki = WikipediaAPIWrapper()

# Show stuff to the screen if there's a prompt
portfolio = 'Apple: 60%, NextEraEnergy: 40%'
if prompt:
    e_research = wiki.run(prompt)
    p_research = wiki.run('Risk Analysis: '+ portfolio)
    event = input_chain.run(event= prompt, event_research= e_research)
    script_output = output_chain.run(input_chain_output=event, portfolio = portfolio, portfolio_research = p_research)

    st.write(event)
    st.write(script_output) 

    with st.expander('Title History'): 
        st.info(title_memory.buffer)

    with st.expander('Script History'): 
        st.info(script_memory.buffer)

    with st.expander('Event Research'): 
        st.info(e_research)

    with st.expander('Portfolio Research'): 
        st.info(p_research)
