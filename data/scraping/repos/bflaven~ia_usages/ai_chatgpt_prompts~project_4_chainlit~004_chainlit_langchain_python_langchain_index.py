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
chainlit run 004_chainlit_langchain_python_langchain_index.py -w



The -w flag tells Chainlit to enable auto-reloading, so you don’t need to restart the server every time you make changes to your application. Your chatbot UI should now be accessible at http://localhost:8000.



# other module
# go to the env

# for chainlit
pip install chainlit

for this example
pip install llama-index

https://docs.chainlit.io/integrations/llama-index

Source: https://docs.chainlit.io/pure-python



If your agent does not have an async implementation, fallback to the sync implementation.

This is the sync implementation.
pip install --upgrade pygpt4all==1.0.1
"""
import os
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
import chainlit as cl

os.environ["OPENAI_API_KEY"] = "YOUR_OPEN_AI_API_KEY"

template = """Question: {question}

Answer: Let's think step by step."""


# @cl.langchain_factory(use_async=False)
# @cl.langchain_factory()
def factory():
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=ChatOpenAI(
        temperature=0, streaming=True))

    return llm_chain
