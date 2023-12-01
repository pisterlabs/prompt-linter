#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
[env]
# Conda Environment
# NO CONDA ENV

conda create --name ai_chatgpt_prompts python=3.9.13
conda info --envs
source activate ai_chatgpt_prompts
conda deactivate
# if needed to remove
conda env remove -n [NAME_OF_THE_CONDA_ENVIRONMENT]


# update conda 
conda update -n base -c defaults conda

# to export requirements
pip freeze > ai_chatgpt_prompts.txt


# to install
pip install -r ai_chatgpt_prompts.txt

[path]
cd /Users/brunoflaven/Documents/03_git/BlogArticlesExamples/ai_chatgpt_prompts/project_3_streamlit/

[file]
streamlit run 002_project_3_python_streamlit_app_chatgpt_api.py

https://stackoverflow.com/questions/65602056/how-to-set-and-access-environment-variables-in-python-file-for-streamlit-app

https://techcommunity.microsoft.com/t5/healthcare-and-life-sciences/integrating-azure-openai-with-streamlit-with-example-source-code/ba-p/3809006

https://github.com/ajitdash/pview/blob/main/explaincode.py


pip install openai
pip install streamlit
pip install python-dotenv

"""

import os
import streamlit as st
from dotenv import load_dotenv
import openai

# Load environment variables from .env file
load_dotenv()
# Get API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# load the key
openai.api_key = OPENAI_API_KEY


# Define Streamlit app layout
st.title("Code Explainer")
language = st.selectbox("Select Language", ["Python", "JavaScript"])
code_input = st.text_area("Enter text to translate")


# Define function to explain code using OpenAI Codex
def explain_code(input_code, language):
    model_engine = "text-davinci-002"  # Change to the desired OpenAI model
    prompt = f"Explain the following {language} code: \n\n{input_code}"
    response = openai.Completion.create(
        engine=model_engine,
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.7,
    )
    return response.choices[0].text


# Temperature and token slider
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.1
)
tokens = st.sidebar.slider(
    "Tokens",
    min_value=64,
    max_value=2048,
    value=256,
    step=64
)
# Define Streamlit app behavior
if st.button("Explain"):
    output_text = explain_code(code_input, language)
    st.text_area("Code Explanation", output_text)
