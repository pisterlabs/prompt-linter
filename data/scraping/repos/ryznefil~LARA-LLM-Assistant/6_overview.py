import os
import pickle
import streamlit as st
import openai
import json
from langchain.llms import OpenAI
from langchain import PromptTemplate
from llama_hub.file.pymu_pdf.base import PyMuPDFReader
# from llama_index import VectorStoreIndex, ServiceContext, LLMPredictor
from llama_index import download_loader

from scipy.io.wavfile import write

# This module is imported so that we can 
# play the converted audio
import os
from datetime import datetime


st.title("LARA Overview")


def load_memory():
    data_folder = 'data/'
    pickle_files = [f for f in os.listdir(data_folder) if f.endswith('outputs.json')]

    context = ''

    for file in pickle_files:
        with open(os.path.join(data_folder, file), 'rb') as f:
            data = json.load(f)
            context += str(data)
            
    # pdf_folder = 'datapdf/'
    # pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]

    # for file in pdf_files:
    #     reader = PyMuPDFReader()
    #     PDFReader = download_loader("PDFReader")
    #     loader = PDFReader()
    #     documents = loader.load_data(file=os.path.join(pdf_folder, file))
    #     # index = VectorStoreIndex.from_documents(documents)
    #     # index.storage_context.persist()
    #     context += documents[0].text

    context += f'/n {datetime.now()}'

    return context

def querry_llm(context):
    prompt_ = """
            **ROLE:**
            - You are a data reconstruction expert. Who specializes in taking data from JSON inputs and formatting them in a human-readable formatted string.
            
            **TASK:**
            - Use the JSON "Context" below to compile markdown output containing relevant user information in a Markdown table format. As shown in the "Examples".

            **Context:**
            {context}

            **Desired Output Format:**
            Output the user information in the following Markdown table format, create multiple tables for different type of infomation. 
            
            **Output Table Example:**
            General Personal Information:
            | <Category>       | <Data>           |
            |------------------|------------------|
            | ...              | ...              |
            
            ###
            
            Output the user information in the following Markdown table format, create multiple tables for different type of infomation:
            """
            
    template = PromptTemplate(template=prompt_, input_variables=["context"])
    
    llm = OpenAI(model_name="gpt-4", temperature=0, max_tokens=300)
    prompt = template.format(context=context)
    
    print("THE PROMPT IS: " + "\n" + str(prompt))
    
    res = llm(prompt)

    return res


####### RUNNING FROM HERE
context = load_memory()
if 'context' not in st.session_state:
    st.session_state.context = context

res = querry_llm(context)
    
st.write(res)