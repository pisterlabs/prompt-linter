#! /usr/bin/env python3

# 2023-05-09
#  ref.: <https://medium.com/the-techlife/using-huggingface-openai-and-cohere-models-with-langchain-db57af14ac5b>
#  the problem with this post is that the code has not been proofread or tested;
#    contains several errors and unneeded imports
# the code here was copied from my own REPL and cleaned up (WLA)

from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.vectorstores import FAISS

import glob

template = """ You are my assistant.
Please provide the best answers to my question with reasoning for why they are correct.
Question: {input} Answer: """

prompt = PromptTemplate(template=template, input_variables=["input"])
model = HuggingFaceHub(
#    repo_id="decapoda-research/llama-7b-hf",
    repo_id="facebook/mbart-large-50",
    model_kwargs={"temperature": 0, "max_length":200}
)

chain = LLMChain(prompt=prompt,llm=model)
hf_embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
my_loader = DirectoryLoader('/Users/band/tmp/workbench/testdir', glob='**/*.txt')
docs = my_loader.load()
text_split = RecursiveCharacterTextSplitter(chunk_size = 700, chunk_overlap = 0)
text = text_split.split_documents(docs)

vectorstore = FAISS.from_documents(text, hf_embeddings)
my_chain = load_qa_with_sources_chain(model, chain_type="refine")
query = "What are the main topics?"
documents = vectorstore.similarity_search(query)
result = my_chain({"input_documents": documents, "question": query})

from pprint import pprint
pprint(result)

