#This script is a Python code for a chatbot that answers questions based on a PDF document. 

#The chatbot uses LangChain library to perform Natural Language Processing (NLP) tasks, 
#OpenAI library for language modeling, and Faiss library for vector indexing and search.

#1)
#The code starts by importing necessary libraries such as os, pathlib, re, fitz, langchain, openai, and faiss. 
#Then, it sets up an environment variable for the OpenAI API key using the os library.

#2)
#Next, the script defines the path to the PDF file and the data storage directory. 
#It loads the PDF file using the fitz library and extracts the text content of each page. 
#Then, it converts the text content into LangChain's Document object and splits it into smaller chunks using the CharacterTextSplitter object.

#3)
#After that, the code uses OpenAIEmbeddings object to convert the text chunks into embeddings,
# and it creates a vector store using the Faiss library. This vector store is used to retrieve relevant text chunks based on a given query.

#4)
#The script then sets up the chat model using the ChatOpenAI object and defines a prompt that includes both the system message and the user message. 
#It also defines a chain that uses the vector store to retrieve relevant text chunks and answers the user's question.

#5)
#Finally, the code defines a function that prints the result of the chatbot's answer, and it uses the chain object to answer two example questions. 
#The last part of the code turns on the debugging mode to see the OpenAI requests made during the chatbot's execution.

###################################################################################################################################################

import os
import pathlib
import re
import fitz
import langchain
import openai
import faiss
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain

# Set up OPEN_API_KEY and necessary variables
os.environ["OPENAI_API_KEY"] = "YOUR-OPENAI-API-KEY-HERE" # Replace this with your key

# Path to the PDF file
PDF_PATH = "/Users/handout.pdf"
DATA_STORE_DIR = "data_store"

# Load documents and split them into chunks for conversion to embeddings
doc = fitz.open(PDF_PATH)
pdf_text = "\n".join([page.get_text("text") for page in doc])

def convert_path_to_doc_url(doc_path):
    return f"pdf://{doc_path}"

# Create a Document object with the PDF text content and metadata
document = Document(
    page_content=pdf_text,
    metadata={"source": convert_path_to_doc_url(PDF_PATH)}
)

# Set up the text splitter to divide the document into smaller chunks
text_splitter = CharacterTextSplitter(separator="\n### ", chunk_size=100, chunk_overlap=20)

# Split the document into smaller chunks for better processing
split_docs = text_splitter.split_documents([document])

# Create Vector Store using OpenAI
embeddings = OpenAIEmbeddings()
vector_store = langchain.FAISS.from_documents(split_docs, embeddings)

# Set up the chat model and specific prompt
system_template = """Use the following pieces of context to answer the users question.
Take note of the sources and include them in the answer in the format: "SOURCES: source1 source2", use "SOURCES" in capital letters regardless of the number of sources.
If you don't know the answer, just say that "I don't know", don't try to make up an answer.
----------------
{summaries}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}")

]
prompt = ChatPromptTemplate.from_messages(messages)

# Set up the chain model and its arguments
chain_type_kwargs = {"prompt": prompt}
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=256)
chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# Function to print the result in a formatted manner
def print_result(result):
    output_text = f"""### Question: 
    {query}
    ### Answer: 
    {result['answer']}
    ### Sources: 
    {result['sources']}
    ### All relevant sources:
    {' '.join(list(set([doc.metadata['source'] for doc in result['source_documents']])))}
    """
    print(output_text)

# Use the chain to query
query = "What is the role of the Executive Committee?"
result = chain(query)
print_result(result)

# Turn on debugging to see the OpenAI requests
import logging
logging.getLogger("openai").setLevel(logging.DEBUG)

query = "What is the role of the Executive Committee?"
result = chain(query)
print_result(result)
