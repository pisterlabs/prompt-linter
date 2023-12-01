#! /usr/bin/env python3
"""
This program generates an OpenAI chat-bot from a directory of files and reads queries from the command line.
Filenames with the following extensions are ignored: ".docx",".jpg",".pdf",".png",".pptx".

Program is terminated by entering "quit", "exit", or "bye" at the query prompt.

code derived from <https://bootcamp.uxdesign.cc/a-step-by-step-guide-to-building-a-chatbot-based-on-your-own-documents-with-gpt-2d550534eea5>
A step-by-step guide to building a chatbot based on your own documents with GPT

Required library: This command is one way to install LlamaIndex and OpenAI Python libraries.

!pip install llama-index

program assumptions:
(1) OPENAI_API_KEY has been set in the shell environment
(2) documents are read from a directory containing text, markdown, and other filess
(3) GPT-index is generated every time the program is run (running cost factor)
(4) generated index is saved as `index.json`, but it is not reused

TODOs:
- save and re-use GPT-index file from previous run (save some time and money)
- maybe save queries and responses to a log file

"""

# set up logging
import logging, os
logging.basicConfig(level=os.environ.get('LOGLEVEL', 'WARNING').upper())

# import needed packages
import glob
from pathlib import Path

from llama_index import (
    GPTVectorStoreIndex,
    LLMPredictor,
    PromptHelper,
    ServiceContext,
    )
from langchain import OpenAI

# define prompt helper properties
max_input_size = 4096
num_output = 256
max_chunk_overlap = 20
prompt_helper = PromptHelper(max_input_size=max_input_size, num_output=num_output, max_chunk_overlap=max_chunk_overlap)

# define LLM
llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-ada-001"))

service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

from llama_index import download_loader
UnstructuredReader = download_loader('UnstructuredReader')
loader = UnstructuredReader()

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
    
    dir_path = str(args.directory)
    logging.info("document file directory: %s", dir_path)

    # Loading from a directory
    logging.debug("Loading from directory %s", dir_path)
    documents = []
    allfiles = [Path(f).as_posix() \
                for f in glob.iglob(f"{dir_path}/**/*.*", recursive=True, include_hidden=False) \
                if Path(f).suffix not in [".docx",".jpg",".pdf",".png",".pptx"]]

    for file_path in allfiles:
        logging.debug("file_path: %s", file_path)
        documents.extend(loader.load_data(file=file_path, split_documents=False))

    logging.debug("how many documents? %s", len(documents))

    # Construct a simple vector index
    index = GPTVectorStoreIndex.from_documents(documents, service_context=service_context)

    # Save your index to a index.json file
    index.save_to_disk('index.json')
    # Load the index from your saved index.json file
#    index = GPTSimpleVectorIndex.load_from_disk('index.json')
    
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
                response = index.query(f"{query!r}")
                print(response)
 
if __name__ == "__main__":
    exit(main())

