import openai
import streamlit as st

from langchain import OpenAI, PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain

def summarizeChat(prompt,text):
    llm = OpenAI(temperature=0)
    # userPrompt = 
    prompt_template = prompt+""":


    {text}


    """
    PROMPT = PromptTemplate( input_variables=["text"], template=prompt_template )
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size = 4000,
        chunk_overlap  = 20,
        length_function = len,
    )
    docs = text_splitter.create_documents([text])
    try:
        chain = load_summarize_chain(llm, chain_type="map_reduce", return_intermediate_steps=True, map_prompt=PROMPT, combine_prompt=PROMPT)
        summary = chain({"input_documents": docs},return_only_outputs=True)
        st.session_state["summary"] = summary['output_text']
    except:
        st.write('There was an error =(')
