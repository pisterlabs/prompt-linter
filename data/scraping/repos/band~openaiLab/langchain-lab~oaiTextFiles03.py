#! /usr/bin/env python3
"""
2023-04-30: update TextFiles01 to use langchain QA use case example.
 - cf. https://python.langchain.com/en/latest/index.html
 - this will replace the use of llama_index ???

This program generates an OpenAI chat-bot from a directory of files and reads queries from the command line.
Filenames with the following extensions are ignored: ".docx",".jpg",".pdf",".png",".pptx".

Program is terminated by entering "quit", "exit", or "bye" at the query prompt.

code derived from <https://bootcamp.uxdesign.cc/a-step-by-step-guide-to-building-a-chatbot-based-on-your-own-documents-with-gpt-2d550534eea5>
A step-by-step guide to building a chatbot based on your own documents with GPT

Required library: This command is one way to install langchain and OpenAI Python libraries.

!pip install langchain
!pip install openai

program assumptions:
(1) OPENAI_API_KEY has been set in the shell environment
(2) documents are read from a directory containing text, markdown, and other filess
(3) index is generated every time the program is run (running cost factor)

TODOs:
- save and re-use index file from previous run (save some time and money)
- maybe save queries and responses to a log file

"""
# set up logging
import logging, os
logging.basicConfig(level=os.environ.get('LOGLEVEL', 'WARNING').upper())

# import needed packages
import glob
from pathlib import Path

from langchain.document_loaders import DirectoryLoader, TextLoader

from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

#os.environ["OPENAI_API_KEY"] = "sk-xxx"

# set up argparse
import argparse
def init_argparse():
    parser = argparse.ArgumentParser(description='Generate OpenAI chat-bot from a directory of text, Markdown and other files.')
    parser.add_argument('--directory', '-d', required=True, help='directory')
    return parser

def main():
    argparser = init_argparse();
    args = argparser.parse_args();
    logging.debug(f"args: {args}")

    # Load the documents
    loader = TextLoader(file_path='/Users/band/tmp/workbench/aFewTextDocs/reviewOfHuntington.txt')

    #creates an object with vectorstoreindexcreator
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":'./_indices'}).from_loaders([loader])

    llm = OpenAI(temperature=0, model='gpt-3.5-turbo')
    # Create a question-answering chain using the index
    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=index.vectorstore.as_retriever(), input_key="question")

    # Query the index
    while True:
        # run a query read from the input
        query = input("enter a query: ")
        match query.split():
            case ["quit" | "-q" | "exit" | "bye"]:
                logging.debug("we quit!")
                quit()
            case _:
                print(f"run this query: {query!r}.")
                response = chain({"question":query})
                print(response)
 
if __name__ == "__main__":
    exit(main())

