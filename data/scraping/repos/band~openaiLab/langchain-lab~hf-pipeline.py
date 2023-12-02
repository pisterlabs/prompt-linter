#! /usr/bin/env python3

# 2023-05-09
#  copied from huggingface-langchain.py
# the code here was copied from my own REPL (WLA)

from langchain import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.vectorstores import FAISS

import glob

model = HuggingFacePipeline.from_model_id(model_id="facebook/mbart-large-50", task="text-generation", model_kwargs={"temperature":0, "max_length":300})

template = """ You are an assistant.
Provide answers to my question with sources as support.
Question: {input} Answer: """

prompt = PromptTemplate(template=template, input_variables=["input"])

chain = LLMChain(prompt=prompt,llm=model)

hf_embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

my_loader = DirectoryLoader('/Users/band/tmp/workbench/testdir', glob='**/*.txt')
docs = my_loader.load()
text_split = RecursiveCharacterTextSplitter(chunk_size = 700, chunk_overlap = 0)
text = text_split.split_documents(docs)

vectorstore = FAISS.from_documents(text, hf_embeddings)

my_chain = load_qa_with_sources_chain(model, chain_type="refine")
query = "What information topics are in this collection?"
documents = vectorstore.similarity_search(query)
result = my_chain({"input_documents": documents, "question": query})

from pprint import pprint
pprint(result)

