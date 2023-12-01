from fastapi import FastAPI
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import tiktoken
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/test")
async def test():
    return {"message": "Hello, World!"}

# Load environment variables from the .env file
load_dotenv(".env", override=True)

# Access the OPENAI_API_KEY environment variable
openai_api_key = os.environ.get("OPENAI_API_KEY")
openai_api_type = os.environ.get("OPENAI_API_TYPE")
openai_api_base = os.environ.get("OPENAI_API_BASE")
openai_api_version = os.environ.get("OPENAI_API_VERSION")

# print(openai_api_key)
# print(openai_api_type)
# print(openai_api_base)
# print(openai_api_version)

class Item(BaseModel):
    key: str

# location of the pdf file/files. 
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# raw_text=get_pdf_text(['docs/RestrictAct.pdf', 'docs/Gandhi.pdf'])
raw_text=get_pdf_text(['docs/Orca.pdf'])

# We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits. 
text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

# Download embeddings from OpenAI
embeddings_model = "CaztonEmbedAda2"
tokenizer = tiktoken.get_encoding("cl100k_base")

# add embeddings model to it and then create a vector store
embeddings = OpenAIEmbeddings(
    deployment = embeddings_model,
    chunk_size = 1)

docsearch = FAISS.from_texts(texts, embeddings)

@app.get("/runFirst")
async def runFirst():
    docsearch = FAISS.from_texts(texts, embeddings)

    with open("docsearch.pkl", "wb") as f:
        pickle.dump(docsearch, f)

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

gpt3_model = "CaztonDavinci3"

chain = load_qa_chain(OpenAI(engine=gpt3_model, temperature=0), chain_type="stuff")

@app.get("/")
async def root():
    return {"message": "Welcome to the question-answering API!"}

@app.post("/qa")
async def get_answer(item: Item):
    query = item.key
    # Perform similarity search using the query
   
    with open("docsearch.pkl", "rb") as f:
        docsearch = pickle.load(f)
        docs = docsearch.similarity_search(query)
    
    # Run the question-answering chain
    response = chain.run(input_documents=docs, question=(query))
    
    # Return the response
    return {"response": response}



