from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import speech_recognition as sr
from fastapi.middleware.cors import CORSMiddleware
import openai
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this with the appropriate origins
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")

with open("openai_key.txt","r") as f:
   openai_key = f.read()
f.close()

with open("pinecone_key.txt","r") as f:
   pinecone_key = f.read()
f.close()

import os
os.environ["OPENAI_API_KEY"] = openai_key


embeddings = OpenAIEmbeddings()


directory = 'data/'

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

documents = load_docs(directory)

def split_docs(documents,chunk_size=1000,chunk_overlap=20):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)

# initialize pinecone
pinecone.init(
    api_key= pinecone_key,  # find at app.pinecone.io
    environment="northamerica-northeast1-gcp"  # next to api key in console
)

index_name = "index"

index = Pinecone.from_documents(docs, embeddings, index_name=index_name)


model_name = "text-davinci-002"
# model_name = "gpt-3.5-turbo"
# model_name = "gpt-4"
llm = OpenAI(model_name=model_name)
chain = load_qa_chain(llm, chain_type="stuff")

def get_similiar_docs(query,k=2,score=False):
  if score:
    similar_docs = index.similarity_search_with_score(query,k=k)
  else:
    similar_docs = index.similarity_search(query,k=k)
  return similar_docs


def get_answer(query):
  similar_docs = get_similiar_docs(query)
  # print(similar_docs)
  answer =  chain.run(input_documents=similar_docs, question=query)
  return  answer



@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/record/audio")
async def record_audio(request: Request):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening....")
        r.pause_threshold = 1
        audio = r.record(source, duration=3)
    try:
        print("Recognizing....")
        query = r.recognize_google(audio, language='en-in')
        print(f"user said: {query}\n")
    except Exception as e:
        print(e)
        query = "None"
    ans =  get_answer(query)
    return {"Answer": ans}
