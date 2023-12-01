import os
import streamlit as st
import pickle
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import UnstructuredURLLoader
from langchain.vectorstores import FAISS
from langchain.llms import GPT4All
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate

loader = UnstructuredURLLoader(urls=[
    "https://www.octoparse.com/blog/how-to-scrape-news",
    "https://proxyscrape.com/blog/web-scraping-for-news-articles-using-python",
    "https://www.zyte.com/learn/what-is-web-scraping/"
])
data = loader.load()

r_splitter = RecursiveCharacterTextSplitter(
    separators = ["\n\n", "\n", " "],  # List of separators based on requirement (defaults to ["\n\n", "\n", " "])
    chunk_size = 1000,  # size of each chunk created
    chunk_overlap  = 100,  # size of  overlap between chunks in order to maintain the context
    length_function = len  # Function to calculate size, currently we are using "len" which denotes length of string however you can pass any token counter)
)

docs = r_splitter.split_documents(data)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vector_index = FAISS.from_documents(docs, embeddings)

# file_path = "vector_index.pkl"
# with open(file_path, "wb") as f:
#   pickle.dump(vector_index, f)

# file_path = "vector_index.pkl"

# if os.path.exists(file_path):
#     with open(file_path, "rb") as f:
#         vector_index = pickle.load(f)

callbacks=[StreamingStdOutCallbackHandler()]

local_path = "D:\Documents\Interview Prep\Code\GPT4All\orca-mini-3b.ggmlv3.q4_0.bin"

llm = GPT4All(model=local_path, callbacks=callbacks, verbose=True)


prompt_template = """
Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). If you don't know the answer, just say that you don't know, don't try to make up an answer.
{summaries}
QUESTION: {question}
SOURCES:
FINAL ANSWER:
"""
doc_prompt_template = """
Content: {page_content}
Source: {source}
"""

DOC_PROMPT = PromptTemplate(
    template=doc_prompt_template, input_variables=["page_content", "source"])

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["summaries", "question"]
)

chain_type_kwargs = {"prompt": PROMPT, "document_prompt": DOC_PROMPT }
chain = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, chain_type="stuff",retriever=vector_index.as_retriever(),chain_type_kwargs=chain_type_kwargs,return_source_documents=True,verbose=True)

query = "What is web scraping?"

answer = chain({"question": query}, return_only_outputs=True)
print(answer["answer"])
print(answer["source_documents"][0].metadata["source"]) # workaround to get the source, as langchain.RetrievalQAWithSourcesChain does not return the source properly.


