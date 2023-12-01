# Databricks notebook source
# DBTITLE 1,Install Gradio Dependencies
# MAGIC %pip install Jinja2==3.0.3 fastapi==0.100.0 uvicorn nest_asyncio databricks-cli gradio==3.37.0 nest_asyncio

# COMMAND ----------

# DBTITLE 1,Install all the libraries for GPU 
# MAGIC %run ./util/install-llm-libraries

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "./util/notebook-config"

# COMMAND ----------

# DBTITLE 1,Optional : Load Ray Dashboard to show cluster Utilisation
# MAGIC %run "./util/install_ray"

# COMMAND ----------

import gradio as gr

import re
import time
import pandas as pd

from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts import PromptTemplate

from util.embeddings import load_vector_db
from util.mptbot import HuggingFacePipelineLocal
from util.QAbot import QABot
from util.DatabricksApp import DatabricksApp

from langchain import LLMChain

# COMMAND ----------

n_documents = 5

# COMMAND ----------

# Retrieve the vector database:
retriever = load_vector_db(config['embedding_model'],
                           config,
                           n_documents = n_documents)


# COMMAND ----------

if config['model_id']  == 'openai':

  # define system-level instructions
  system_message_prompt = SystemMessagePromptTemplate.from_template(config['system_message_template'])

  # define human-driven instructions
  human_message_prompt = HumanMessagePromptTemplate.from_template(config['human_message_template'])

  # combine instructions into a single prompt
  chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

  # define model to respond to prompt
  llm = ChatOpenAI(model_name=config['openai_chat_model'], temperature=config['temperature'])

else:

  # define system-level instructions
  system_message_prompt = SystemMessagePromptTemplate.from_template(config['template'])
  chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])

  # define model to respond to prompt
  llm = HuggingFacePipelineLocal.from_model_id(
    model_id=config['model_id'],
    task="text-generation",
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
    return "", chat_history , info['vector_doc']

with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown(
        """
        # Policy Retrieval QA using Falcon-30b chat variant
        The current version FAISS vector store to Fetch the most relevant paragraph's to create the bot
        """)
    with gr.Row():
        with gr.Column():
            chatbot = gr.Chatbot()
            msg = gr.Textbox(label="Ask your Question")
            clear = gr.ClearButton([msg, chatbot])
        with gr.Column():
            raw_text = gr.Textbox(label="Document from which the answer was generated",scale=50)
    with gr.Row():
      examples = gr.Examples(examples=["What is the duration for the policy with the start and end date", "What is the limit of misfueling cover",
                                       "what does the policy say about loss of car keys","what is the vehicle age covered by the policy","what is the name of the policy holder"],
                        inputs=[msg])
    msg.submit(respond, [msg, chatbot], [msg, chatbot,raw_text])

# COMMAND ----------

 dbx_app = DatabricksApp(8098)
dbx_app.mount_gradio_app(gr.routes.App.create_app(demo))

# COMMAND ----------

import nest_asyncio
nest_asyncio.apply()
dbx_app.run()

# COMMAND ----------

# kill the gradio process
# ! kill -9  $(ps aux | grep 'databricks/python_shell/scripts/db_ipykernel_launcher.py' | awk '{print $2}')

# COMMAND ----------



# COMMAND ----------


