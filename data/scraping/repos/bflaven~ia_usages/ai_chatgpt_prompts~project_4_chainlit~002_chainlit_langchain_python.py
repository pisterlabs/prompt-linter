#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
[env]
# Conda Environment
# NO CONDA ENV
conda create --name chainlit_python python=3.9.13
conda info --envs
source activate chainlit_python
conda deactivate
# if needed to remove
conda env remove -n [NAME_OF_THE_CONDA_ENVIRONMENT]
# examples
conda env remove -n po_launcher_e2e_cypress
conda env remove -n parse_website

# update conda 
conda update -n base -c defaults conda

# to export requirements
pip freeze > requirements_chainlit_python.txt


# to install
pip install -r requirements_chainlit_python.txt



[path]
cd /Users/brunoflaven/Documents/03_git/BlogArticlesExamples/ai_chatgpt_prompts/project_4_chainlit/


[file]
chainlit run 002_chainlit_langchain_python.py -w

The -w flag tells Chainlit to enable auto-reloading, so you don’t need to restart the server every time you make changes to your application. Your chatbot UI should now be accessible at http://localhost:8000.



# other module
# go to the env

# for chainlit
pip install chainlit




Source: https://docs.chainlit.io/pure-python

"""

 
import os
from langchain import PromptTemplate, OpenAI, LLMChain
import chainlit as cl

# PAID ONE DO NOT DISPLAY
# add your own api key for open ai
os.environ["OPENAI_API_KEY"] = 'OPENAI_API_KEY'

template = """Question: {question}

Answer: Let's think step by step."""


@cl.langchain_factory
def factory():
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=OpenAI(
        temperature=0), verbose=True)

    return llm_chain
