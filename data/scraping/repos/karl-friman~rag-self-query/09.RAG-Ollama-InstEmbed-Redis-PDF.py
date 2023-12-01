# -*- coding: utf-8 -*-

from langchain.llms import Ollama

llm = Ollama(model="openhermes2-mistral", temperature=0.1)

import os

# from langchain.vectorstores import Chroma
from langchain.vectorstores.redis import Redis
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader


from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings

"""## Load multiple and process documents"""

loader = DirectoryLoader("./whitepapers/", glob="./*.pdf", loader_cls=PyPDFLoader)

documents = loader.load()

len(documents)

# splitting the text into
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(documents)

from langchain.embeddings import HuggingFaceInstructEmbeddings

# instructor_embeddings = HuggingFaceInstructEmbeddings(
#     model_name="hkunlp/instructor-xl", model_kwargs={"device": "cuda"}
# )
instructor_embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-xl", model_kwargs={"device": "cpu"}
)

persist_directory = "db"

embedding = instructor_embeddings

# vectordb = Chroma.from_documents(
#     documents=texts, embedding=embedding, persist_directory=persist_directory
# )
vectordb = Redis.from_documents(
    documents=texts,
    embedding=embedding,
    redis_url="redis://localhost:6379",
)

"""## Make a retriever"""

retriever = vectordb.as_retriever(search_kwargs={"k": 5})

"""## Make a chain"""

# create the chain to answer questions
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True
)

## Cite sources

import textwrap


def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split("\n")

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = "\n".join(wrapped_lines)

    return wrapped_text


def process_llm_response(llm_response):
    print(wrap_text_preserve_newlines(llm_response["result"]))
    print("\n\nSources:")
    for source in llm_response["source_documents"]:
        print(source.metadata["source"])


query = "What is toolformer?"
print(query)
llm_response = qa_chain(query)
process_llm_response(llm_response)

query = "What tools can be used with toolformer?"
print(query)
llm_response = qa_chain(query)
process_llm_response(llm_response)

query = "How many examples do we need to provide for each tool?"
print(query)
llm_response = qa_chain(query)
process_llm_response(llm_response)
