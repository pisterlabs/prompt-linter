# Databricks notebook source
# DBTITLE 1,Install Gradio Dependencies
# MAGIC %pip install Jinja2==3.0.3 fastapi==0.100.0 uvicorn nest_asyncio databricks-cli gradio==3.37.0 nest_asyncio

# COMMAND ----------

# MAGIC %run "./util/install-prep-libraries"

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./util/notebook-config"

# COMMAND ----------

import gradio as gr

import re
import time
import pandas as pd

from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts import PromptTemplate

from util.embeddings import load_vector_db
from util.mptbot import HuggingFacePipelineLocal,TGILocalPipeline
from util.qabot import QABot
from langchain.chat_models import ChatOpenAI
from util.DatabricksApp import DatabricksApp

from langchain import LLMChain

# COMMAND ----------

n_documents = 10

# COMMAND ----------

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings,HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS


def load_vector_db(embeddings_model = 'intfloat/e5-large-v2',
                   config = None,
                   n_documents = 5):
  '''
  Function to retrieve the vector store created
  '''
  if config['model_id'] == 'openai' :
    embeddings = OpenAIEmbeddings(model=config['embedding_model'])
  else:
    if "instructor" in config['embedding_model']:
      embeddings = HuggingFaceInstructEmbeddings(model_name= config['embedding_model'])
    else:
      embeddings = HuggingFaceEmbeddings(model_name= config['embedding_model'])
  vector_store = FAISS.load_local(embeddings=embeddings, folder_path=config['vector_store_path'])
  retriever = vector_store.as_retriever(search_kwargs={'k': n_documents}) # configure retrieval mechanism
  return retriever


# COMMAND ----------

# Retrieve the vector database:
retriever = load_vector_db(config['embedding_model'],
                           config,
                           n_documents = n_documents)


# COMMAND ----------

# define system-level instructions
system_message_prompt = SystemMessagePromptTemplate.from_template(config['template'])
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

if config['model_id']  == 'openai':

  # define model to respond to prompt
  llm = ChatOpenAI(model_name=config['openai_chat_model'], temperature=config['temperature'])

else:
  # define model to respond to prompt
  llm = TGILocalPipeline.from_model_id(
    model_id=config['model_id'],
    model_kwargs =config['model_kwargs'],
    pipeline_kwargs= config['pipeline_kwargs'])

# Instatiate the QABot
qabot = QABot(llm, retriever, chat_prompt)

# COMMAND ----------

# DBTITLE 1,Create the Gradio Template
def respond(question, chat_history):
    print(question)
    info = qabot.get_answer(question)
    chat_history.append((question,info['answer']))
    return "", chat_history , info['vector_doc'], info['source']

with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown(
        f"""
        # Policy Retrieval QA using {config['model_id']}
        The current version FAISS vector store to Fetch the most relevant paragraph's to create the bot
        """)
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="Ask your Question")
            clear = gr.ClearButton([msg, chatbot])
        with gr.Column():
            raw_text = gr.Textbox(label="Document from which the answer was generated",scale=50)
            raw_source = gr.Textbox(label="Source of the Document",scale=1)
    with gr.Row():
      examples = gr.Examples(examples=["what is limit of the misfueling cost covered in the policy?", "what happens if I lose my keys?","what is the duration for the policy bought by the policy holder mentioned in the policy schedule / Validation schedule","What is the maximum Age of a Vehicle the insurance covers?"],
                        inputs=[msg])
    msg.submit(respond, [msg, chatbot], [msg, chatbot,raw_text,raw_source])

    # 
# ""
# ""

# COMMAND ----------

 dbx_app = DatabricksApp(8098)
dbx_app.mount_gradio_app(gr.routes.App.create_app(demo))

# COMMAND ----------

import nest_asyncio
nest_asyncio.apply()
dbx_app.run()

# COMMAND ----------

# kill the gradio process
! kill -9  $(ps aux | grep 'databricks/python_shell/scripts/db_ipykernel_launcher.py' | awk '{print $2}')
