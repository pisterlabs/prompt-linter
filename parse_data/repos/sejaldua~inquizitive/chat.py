from langchain.embeddings.openai import OpenAIEmbeddings
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders import UnstructuredPowerPointLoader
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

import os
import nltk
import constants
import logging
import streamlit as st
import tempfile
import re

# Initialize logging with the specified configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(constants.LOGS_FILE),
        logging.StreamHandler(),
    ],
)
LOGGER = logging.getLogger(__name__)

# load documents from the specified directory using a DirectoryLoader object

def load_documents(files=[]):
    if len(files) == 0:
        loader = DirectoryLoader(constants.FILE_DIR)
        documents = loader.load()
    else:
        documents = []
        for f in files:
            print(f)
            temp_dir = tempfile.TemporaryDirectory()
            temp_filepath = os.path.join(temp_dir.name, f.name)
            with open(temp_filepath, "wb") as fout:
                fout.write(f.read())
            fname = f.name
            print(fname)
            if fname.endswith('.pdf'):
                loader = PyPDFLoader(temp_filepath)
                documents.extend(loader.load())
            elif fname.endswith('.doc'):
                loader = Docx2txtLoader(temp_filepath)
                documents.extend(loader.load())
            elif fname.endswith('.txt'):
                loader = TextLoader(temp_filepath)
                documents.extend(loader.load())
            elif fname.endswith('.md'):
                loader = UnstructuredMarkdownLoader(temp_filepath)
                documents.extend(loader.load())
            elif fname.endswith('.ppt'):
                loader = UnstructuredPowerPointLoader(temp_filepath)
                documents.extend(loader.load())

    # print(documents[0].page_content)
    text = " ".join([re.sub('\s+', ' ', d.page_content) for d in documents])
    return text, documents

def create_doc_embeddings(documents) -> any:
    
    # split the text to chunks of of size 1000
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)

    # create a vector store from the chunks using an OpenAIEmbeddings object and a Chroma object
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_APIKEY'])
    docsearch = Chroma.from_documents(texts, embeddings)
    return docsearch

# Define answer generation function
def answer(prompt: str, docsearch, persist_directory: str = constants.PERSIST_DIR) -> str:
    
    # Log a message indicating that the function has started
    LOGGER.info(f"Start answering based on prompt: {prompt}.")
    
    # Create a prompt template using a template from the config module and input variables
    # representing the context and question.
    prompt_template = PromptTemplate(template=constants.prompt_template, input_variables=["context", "question"])
    
    # Load a QA chain using an OpenAI object, a chain type, and a prompt template.
    doc_chain = load_qa_chain(
        llm=OpenAI(
            openai_api_key = os.environ['OPENAI_APIKEY'],
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=300,
        ),
        chain_type="stuff",
        prompt=prompt_template,
    )
    
    # Log a message indicating the number of chunks to be considered when answering the user's query.
    LOGGER.info(f"The top {constants.k} chunks are considered to answer the user's query.")
    
    # Create a VectorDBQA object using a vector store, a QA chain, and a number of chunks to consider.
    qa = VectorDBQA(vectorstore=docsearch, combine_documents_chain=doc_chain, k=constants.k)
    
    # Call the VectorDBQA object to generate an answer to the prompt.
    result = qa({"query": prompt})
    answer = result["result"]
    
    # Log a message indicating the answer that was generated
    LOGGER.info(f"The returned answer is: {answer}")
    
    # Log a message indicating that the function has finished and return the answer.
    LOGGER.info(f"Answering process completed.")
    return answer
