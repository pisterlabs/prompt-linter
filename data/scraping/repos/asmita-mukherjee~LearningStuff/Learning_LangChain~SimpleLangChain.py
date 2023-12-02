import streamlit as st
from getpass import getpass
import os
from langchain import HuggingFaceHub

# HUGGINGFACEHUB_API_TOKEN = getpass()
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN

st.title("Chat With OpenSource Models 🤗")

with st.sidebar:

    huggingface_token_api = st.text_input("Enter your hugging face api token")

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = huggingface_token_api

    model_option = st.selectbox("Select the model to chat",('Flan, by Google','Dolly, by Databricks',
                                                            'Camel, by Writer','XGen, by Salesforce',
                                                            'Falcon, by Technology Innovation Institute (TII)'))
    temperature = st.text_input("Enter the temperature of the model")

    max_length = st.text_input("Enter the max length of the responses")

def generate_response(input_prompt,model_option):

    if model_option == 'Flan, by Google':
        repo_id = "google/flan-t5-xxl"
    elif model_option == "Dolly, by Databricks":
        repo_id = "databricks/dolly-v2-3b"
    elif model_option == 'Camel, by Writer':
        repo_id = "Writer/camel-5b-hf"
    elif model_option == "XGen, by Salesforce":
        repo_id = "Salesforce/xgen-7b-8k-base"
    else:
        repo_id = "tiiuae/falcon-40b"

    llm = HuggingFaceHub(repo_id=repo_id,model_kwargs={"temperature":float(temperature),"max_length":int(max_length)})

    #this will print the response from the llm
    st.info(llm(input_prompt))


with st.form('Chat Form'):
    text = st.text_area("Enter Prompt","Teach me about LLMs")
    submitted = st.form_submit_button("Submit question")
    if submitted:
        with st.spinner("Please wait for response"):
            generate_response(text,model_option)


