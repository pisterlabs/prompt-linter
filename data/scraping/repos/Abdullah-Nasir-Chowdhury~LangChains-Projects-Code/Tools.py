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
from langchain.utilities import WikipediaAPIWrapper

st.title('YouTube GPT Generator')
prompt = st.text_input('Plug in your prompt here!')

title_template = PromptTemplate(input_variables=['topic'], 
                                template='Write me a youtube video title about {topic}'
                                )
script_template = PromptTemplate(input_variables=['title', 'wikipedia_research'],
                                 template='Write me a script based on the video title: {title} \
                                     while leveraging this wikipedia research {wikipedia_research}'
                                 )


title_memory = ConversationBufferMemory(input_key='topic', 
                                  memory_key='chat_history')

script_memory = ConversationBufferMemory(input_key='title',
                                         memory_key='chat_history')

llm = OpenAI(temperature=0)

title_chain = LLMChain(llm=llm, 
                       prompt=title_template, 
                       verbose=True, 
                       output_key='title',
                       memory=title_memory)

script_chain = LLMChain(llm=llm, 
                        prompt=script_template, 
                        verbose=True, 
                        output_key='script',
                        memory=script_memory)

wiki = WikipediaAPIWrapper()

if st.button(label='Generate'):
    if prompt:
        title = title_chain.run(prompt)
        wiki_research = wiki.run(prompt)
        script = script_chain.run(title=title, wikipedia_research=wiki_research)
        
        st.write(title)
        st.write(script)
        
        with st.expander('Title history'):
            st.info(title_memory.buffer)
        with st.expander('Script history'):
            st.info(script_memory.buffer)
        with st.expander('Wikipedia Research'):
            st.info(wiki_research)
            
    else:
        st.write('Please enter a prompt!')
        
        