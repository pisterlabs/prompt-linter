# -*- coding: utf-8 -*-
import langchain
import pinecone
import openai
import tiktoken
import nest_asyncio
import os
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = "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"

# initialize pinecone
pinecone.init(
    api_key="96e01bcd-0886-4c2d-8c3b-75e9fab99572",  # find at app.pinecone.io
    environment="us-west4-gcp-free"  # next to api key in console
)

nest_asyncio.apply()

# loader = SitemapLoader(
#     "https://ind.nl/sitemap.xml",
#     filter_urls=["https://ind.nl/en"]
# )
# docs = loader.load()
with open("doc.txt", "r") as f:
    docs = f.read()

# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size = 1200,
#     chunk_overlap  = 200,
#     length_function = len,
# )

# docs_chunks = text_splitter.split_documents(docs)

# chunk_size = 1200
# chunk_overlap = 200

# docs_chunks = []
# for i in range(0, len(docs), chunk_size):
#     chunk = docs[i:i + chunk_size]
#     docs_chunks.append(chunk)

embeddings = OpenAIEmbeddings()
index_name = "chatbot-index"
# docsearch = Pinecone.from_documents(docs_chunks, embeddings, index_name=index_name)
docsearch = Pinecone.from_existing_index(index_name=index_name, embedding=embeddings)

# query = "I run a black-owned bookstore in Brookline, MA and I would like to expand my inventory and networking outreach. I am interested in submitting a business proposal to local university in order to fulfil my needs. Approximately how long does the business proposal process take?"
# docs = docsearch.similarity_search(query)
# print(docs[0])

qa_with_sources = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(), return_source_documents=True)

# query = "I run a black-owned bookstore in Brookline, MA and I would like to expand my inventory and networking outreach. I am interested in submitting a business proposal to local university in order to fulfil my needs. Approximately how long does the business proposal process take at MIT?"
# result = qa_with_sources({"query": query})
# print(result["result"])

# query = "tell me more"
# result = qa_with_sources({"query": query})
# result["result"]

# query = "What are some of the certifications that I can obtain as a black business owner?"
# result = qa_with_sources({"query": query})
# result["result"]

# query = "Who is the POC for business proposal at MIT?"
# result = qa_with_sources({"query": query})
# result["result"]

# while(True):
#   query = input()
#   result = qa_with_sources({"query": query})
#   print(result["result"])

"""Output source documents that were found for the query"""

# result["source_documents"]

