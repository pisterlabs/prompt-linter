import sys

from dotenv import load_dotenv
load_dotenv()

from langchain.llms import OpenAI

from langchain.indexes import VectorstoreIndexCreator
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

import openai
import os

openai.api_base = os.environ["OPENAI_API_BASE"]
openai.api_key = os.environ["OPENAI_API_KEY"]

os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.langchain.plus"
os.environ["LANGCHAIN_SESSION"] = os.environ["LANGCHAIN_SESSION"]

target_file = "./earning_reports/AAPL-89728-report.tsv"

def create_retriever(target_file):
    loader = CSVLoader(target_file, csv_args={ 'delimiter': '\t' })
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=2048, chunk_overlap=0
    )
    docs = loader.load_and_split(text_splitter=text_splitter)
    embeddings = OpenAIEmbeddings(
        # model="llama2"
    )
    db = Chroma.from_documents(docs, embeddings)
    return db.as_retriever()

def create_qa_retrival_chain():
    foo_retriever = create_retriever(target_file)
    llm = OpenAI(
        temperature=0, 
        # model_name="llam2", 
        max_tokens=2047,
        request_timeout=240,
    )
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=foo_retriever
    )
    return qa

def main():    
    if len(sys.argv) < 2:
        print("Usage: python retrivalQA.py file_name")
        return

    argument = sys.argv[1]
    print(f"Reading question list from: {argument}")    
    
    print('Loading LLM from', openai.api_base)
    retrival_chain = create_qa_retrival_chain()

    with open(argument, 'r') as file:
        for line in file:
            # Remove the newline character at the end of each line
            line = line.strip()
            if line == '':
                continue
            print('-' * 80)
            print("Questions :", line)
            response = retrival_chain.run(line)
            print("Answer :", response)

if __name__ == '__main__' : 
    main()