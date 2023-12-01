#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corpusaige is a Python tool (and utility library) enabling AI-powered systems analysis 
through deep exploration and understanding of comprehensive document sets and source code.
@copyright: Copyright © 2023 Iwan van der Kleijn
@license: MIT
"""

# Import necessary modules

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader

def init_db():
    # Print number of md files in directory
    book_loader = DirectoryLoader('./test-case/book/src/', glob="./*.md")
    book = book_loader.load()
    print(f'Rust book: {len(book)} pages')

    # Print number of md files in directories
    rbe_loader = DirectoryLoader('./rust-by-example/src/', glob="./**/*.md")
    rbe = rbe_loader.load()
    print(f'Rust by Example: {len(rbe)} pages')

    text_splitter = RecursiveCharacterTextSplitter (chunk_size=1000, chunk_overlap=200)

    chunks = text_splitter.split_documents(book)
    chunks.extend(text_splitter.split_documents(rbe))

    #create directory
    persist_directory = './db'

    # OpenAI embeddings
    embedding = OpenAIEmbeddings()

    vectordb = Chroma.from_documents(documents=chunks,
                                    embedding=embedding,
                                    persist_directory=persist_directory)

    vectordb.persist()

#copy rust-test-case-corpus.ini to corpus.ini in current directory 
def copy_ini():
    import shutil
    shutil.copyfile('./rust-test-case-corpus.ini', './corpus.ini')
    
#init submodules in project, i.e. rust-by-example and rust-book
def init_submodules():
    import subprocess
    subprocess.call(['git', 'submodule', 'update', '--init'])

# Cite sources
def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
        
def test_connection():
    # Test connection to OpenAI API & vector db
    persist_directory = 'db'

    # OpenAI embeddings
    embedding = OpenAIEmbeddings()
    vectordb = Chroma(persist_directory=persist_directory,
                   embedding_function=embedding)
    
    # Create retriever
    retriever = vectordb.as_retriever()
    # create the chain to answer questions
    qa_chain = RetrievalQA.from_chain_type(llm=OpenAI(),
                                  chain_type="stuff",
                                  retriever=retriever,
                                  return_source_documents=True,
                                  verbose=True)
    
    query = "How can I read a file in Rust?"
    llm_response = qa_chain(query)
    process_llm_response(llm_response)
    
copy_ini()
init_submodules()
init_db()
test_connection()
