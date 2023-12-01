"""
Converse with your CSV file through a local web app

Usage in command prompt: 
streamlit run wikipedia_connection.py
"""


import os
from dotenv import load_dotenv

import streamlit as st

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper


def main():

    # Load API Key
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set your OPENAI_API_KEY in the .env file")

    # Initialize LLM
    llm = ChatOpenAI(temperature=0.9)

    # create memory
    title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
    script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

    # create prompt templates
    title_template = PromptTemplate(input_variables=['topic'], template="Write me a funny youtube video title about {topic}")
    script_template = PromptTemplate(input_variables=['title', 'wikipedia_research'],
                                    template="Give me a short paragraph summary of a funny youtube video based on this title: {title} while leveraging this wikipedia research: {wikipedia_research}.")
    
    # set up chains
    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
    script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

    # create wikipedia api wrapper
    wiki = WikipediaAPIWrapper()

    # Define Streamlit interface
    st.title('Wiki Connection Demo 💥')
    prompt = st.text_input('Input a subject to write a funny youtube video based on wikipedia research:')

    # Run chains and display results
    if prompt:
        title = title_chain.run(prompt)
        wiki_research = wiki.run(prompt)
        script = script_chain.run(title=title,wikipedia_research=wiki_research)

        st.write(title)
        st.write(script)

        # show memory in web app
        with st.expander('Title history'):
            st.info(title_memory.buffer)

        with st.expander('Script history'):
            st.info(script_memory.buffer)
        
        with st.expander('Wikipedia research'):
            st.info(wiki_research)

if __name__ == "__main__":
    main()