# Databricks notebook source
# MAGIC %sh
# MAGIC pip install pypdf faiss-cpu langchain tiktoken openai -q

# COMMAND ----------

from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("science.aba6500.pdf")
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
  You are a reader of scientific research papers and your job is to provide the best answer based on the content in a given paper. 
  Use only information in the following paragraphs to answer the question at the end. If you do not know, say that you do not know.

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
  similar_docs = get_similar_docs(question, similar_doc_count=4)
  result = qa_chain({"input_documents": similar_docs, "question": question})
  return result['output_text']

# COMMAND ----------

answer_question("What are the cell types mentioned in this text?")
