from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import openai

print (f'You have {len(data)} document(s) in your data')

print (f'You have {len(data)} document(s) in your data')

print (f'Now you have {len(texts)} documents')


from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

embeddings = OpenAIEmbeddings(openai_api_key="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")
embeddings = OpenAIEmbeddings(
    openai_api_key="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")

openai.api_key = "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"


PINECONE_API_KEY = '2f1f9a16-8e97-4485-b643-bbcd3618570a'
PINECONE_ENVIRONMENT = 'us-west1-gcp-free'

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)


index = pinecone.Index('wing-sandbox')
index.delete(delete_all=True)
docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name="wing-sandbox")

query = "What are examples of good data science teams?"
docs = docsearch.similarity_search(query)

print(docs[0].page_content[:450])


from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

llm = OpenAI(temperature=0, openai_api_key="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")
chain = load_qa_chain(llm, chain_type="stuff")

query = "What is BYOC?"
docs = docsearch.similarity_search(query)

print(chain.run(input_documents=docs, question=query))
