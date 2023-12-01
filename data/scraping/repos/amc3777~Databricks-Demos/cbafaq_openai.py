# Databricks notebook source
# MAGIC %sh
# MAGIC wget -O cba.pdf https://imgix.cosmicjs.com/25da5eb0-15eb-11ee-b5b3-fbd321202bdf-Final-2023-NBA-Collective-Bargaining-Agreement-6-28-23.pdf

# COMMAND ----------

# MAGIC %sh
# MAGIC pip install pypdf faiss-cpu langchain transformers sentence_transformers tiktoken openai -q

# COMMAND ----------

from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("cba.pdf")
pages = loader.load_and_split()

# COMMAND ----------

import os

os.environ["OPENAI_API_KEY"] = ''

# COMMAND ----------

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

embedding_model = OpenAIEmbeddings(model='text-embedding-ada-002')

faiss_index = FAISS.from_documents(pages, embedding_model)

# COMMAND ----------

def get_similar_docs(question, similar_doc_count):
  return faiss_index.similarity_search(question, k=similar_doc_count)

# COMMAND ----------

from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
from langchain.chains.question_answering import load_qa_chain

def build_qa_chain():

  template = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

  Instruction: 
  You are an NBA salary cap expert and your job is to provide the best answer based on the latest NBA collective bargaining agreement. 
  Use only information in the following paragraphs to answer the question at the end. Explain the answer with reference to these paragraphs. If you do not know, say that you do not know.

  {context}
 
  Question: {question}

  Response:
  """
  prompt = PromptTemplate(input_variables=['context', 'question'], template=template)

  llm = OpenAI(model_name="text-davinci-003")

  return load_qa_chain(llm=llm, chain_type="stuff", prompt=prompt)

qa_chain = build_qa_chain()

# COMMAND ----------

def answer_question(question):
  similar_docs = get_similar_docs(question, similar_doc_count=5)
  result = qa_chain({"input_documents": similar_docs, "question": question})
  return result['output_text']

# COMMAND ----------

answer_question("What is the first year salary for a non-taxpayer mid-level exception contract when the salary cap is $134 million?")

# COMMAND ----------

import mlflow

with mlflow.start_run():
    logged_model = mlflow.langchain.log_model(qa_chain, "langchain_model")

# COMMAND ----------

import numpy as np
import faiss

chunk = faiss.serialize_index(faiss_index)
np.save("index.npy", chunk)
index3 = faiss.deserialize_index(np.load("index.npy"))
