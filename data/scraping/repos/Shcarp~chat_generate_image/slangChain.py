import logging
import os
import sys
from dotenv import load_dotenv
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

from langchain.chains import RetrievalQA

from langchain.chat_models import ChatOpenAI

load_dotenv()

loader = WebBaseLoader(['https://stable-diffusion-art.com/prompt-guide/#Anatomy_of_a_good_prompt'])

text_splitter = RecursiveCharacterTextSplitter(
    separators=['Developing', 'comments'],
    chunk_size = 1000,
    chunk_overlap  = 20,
)

data = loader.load_and_split(text_splitter=text_splitter)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap  = 20,
)

texts = text_splitter.split_documents([data[1]])

embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

docsearch = Chroma.from_documents(texts, embeddings)

query = "What is stable diffusion of a good prompt?"

docs = docsearch.similarity_search(query)


chatllm = ChatOpenAI(model_name='gpt-4')

chain3 = RetrievalQA.from_chain_type(llm=chatllm, chain_type='map_reduce', retriever=docsearch.as_retriever())
res = chain3.run(query)

print(res)
