# Bring in deps
import os 
from apikey import apikey 

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('Explain Like I am 5')
prompt = st.text_input('What would you like to learn today?') 

# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='As an expert in {topic}, generate one line that explains the crux of the matter to me.'
)

script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'], 
    template='Explain the crux of the topic TITLE: {title} like I am a novice, while leveraging this wikipedia research:{wikipedia_research} '
)

# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


# Llms
llm = OpenAI(temperature=0.9) 
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

wiki = WikipediaAPIWrapper()

# Show stuff to the screen if there's a prompt
if prompt: 
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt) 
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title) 
    st.write(script) 

    with st.expander('Topic History'): 
        st.info(title_memory.buffer)

    with st.expander('Explanation History'): 
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'): 
        st.info(wiki_research)