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
chainlit run 005_chainlit_langchain_python_llama_index.py -w



The -w flag tells Chainlit to enable auto-reloading, so you don’t need to restart the server every time you make changes to your application. Your chatbot UI should now be accessible at http://localhost:8000.



# other module
# go to the env

# for chainlit
pip install chainlit

for this example
pip install llama-index

https://docs.chainlit.io/integrations/llama-index

Source: https://docs.chainlit.io/pure-python


from llama_index import VectorStoreIndex, SimpleDirectoryReader
documents = SimpleDirectoryReader('data').load_data()
index = VectorStoreIndex.from_documents(documents)


"""
import os
from llama_index.callbacks.base import CallbackManager
from llama_index import (
    LLMPredictor,
    ServiceContext,
    StorageContext,
    SimpleDirectoryReader,
    load_index_from_storage,
)
from langchain.chat_models import ChatOpenAI
import openai

import chainlit as cl

openai.api_key = os.environ.get("OPENAI_API_KEY")

try:
    # Rebuild the storage context
    storage_context = StorageContext.from_defaults(persist_dir="./storage")
    # Load the index
    index = load_index_from_storage(storage_context)
except:
    # Storage not found; create a new one
    from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, VectorStoreIndex, SimpleDirectoryReader

    documents = SimpleDirectoryReader("./data").load_data()
    index = GPTVectorStoreIndex.from_documents(documents)
    index.storage_context.persist()


@cl.llama_index_factory()
def factory():
    llm_predictor = LLMPredictor(
        llm=ChatOpenAI(
            temperature=0,
            model_name="gpt-3.5-turbo",
            streaming=True,
        ),
    )
    service_context = ServiceContext.from_defaults(
        llm_predictor=llm_predictor,
        chunk_size=512,
        callback_manager=CallbackManager([cl.LlamaIndexCallbackHandler()]),
    )
    query_engine = index.as_query_engine(
        service_context=service_context,
        streaming=True,
    )

    return query_engine
