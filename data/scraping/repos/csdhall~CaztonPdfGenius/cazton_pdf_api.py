from typing import Annotated
from fastapi import FastAPI, File, Request, UploadFile
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Item(BaseModel):
    key: str

@app.post("/items/")
async def create_item(item: Item):
    return item

load_dotenv()

openai_api_key = os.environ.get("OPENAI_API_KEY")

@app.post('/qa')
async def run_qa(item: Item):
    question = item.key
    print(question)
    # location of the pdf file/files.
    reader = PdfReader('docs/RestrictAct.pdf')

    reader

    # read data from the file and put them into a variable called raw_text
    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text
        print(question)
        embeddings = OpenAIEmbeddings()
        print(embeddings)
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)
        docsearch = FAISS.from_texts(texts, embeddings)
        chain = load_qa_chain(OpenAI(), chain_type="stuff")
        docs = docsearch.similarity_search(question)
        response = chain.run(input_documents=docs, question=question)
        return {"response": response}

    # uvicorn cazton_pdf_api:app --reload
