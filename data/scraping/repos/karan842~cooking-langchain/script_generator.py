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
st.set_page_config(page_title='YouTube Script Generator')

st.title('📽️📄YouTube Script Generator')
st.markdown('#### using Langchain🦜🔗 & GPT-3🧠💭')
st.success('Generate YouTube or any other type of script by giving just a topic name.')
prompt = st.text_input('Plug in your idea here👇')


# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'],
    template = 'write me a youtube view title about {topic}'
)

script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'],
    template = '''write me a youtube video script based on this title TITLE: {title}
    while levaraging this wikipedia research: {wikipedia_research}'''
)


# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# LLMS
llm = OpenAI(temperature=0.9)

# Chains
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True,
                       output_key='title', memory=title_memory)


script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, 
                        output_key='script',memory=script_memory)

# sequential_chain = SequentialChain(chains=[title_chain, script_chain], 
#                                    input_variables=['topic'],
#                                    output_variables=['title', 'script'],
#                                    verbose=True)

# WIKIPEDDIA 
wiki = WikipediaAPIWrapper()

# show stuff on screen
if prompt:
    # response = sequential_chain({'topic':prompt})
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)
    
    # st.write(response['title'])
    # st.write(response['script'])
    
    st.write(title)
    st.write(script)
    
    with st.expander('Title History'):
        st.info(title_memory.buffer)
        
    with st.expander('Script Research'):
        st.info(script_memory.buffer)
        
    with st.expander('Wikipedia Research'):
        st.info(wiki_research)