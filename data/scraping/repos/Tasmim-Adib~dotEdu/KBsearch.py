from fastapi import FastAPI

import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain.prompts import PromptTemplate

from .EVstore import initializePinecone, existingVectorstore, newDataLoad, \
     extractDocMetaData, createIndex, testing, addToIndex, pinecone

from fastapi.middleware.cors import CORSMiddleware


load_dotenv()

os.environ["PINECONE_API_KEY"] = os.environ.get('pinecone_apikey')
os.environ["PINECONE_ENV"] = os.environ.get('pinecode_env')
os.environ["OPENAI_API_KEY"] = os.environ.get('openAI_apikey')


embeddings = OpenAIEmbeddings()
directory = "app/data"
index_name = "kazinazrul"

# print(len(docs))

initializePinecone()
vectorstore = existingVectorstore(index_name)



def get_similiar_docs(query, k=5, score=False):
  if score:
    similar_docs = vectorstore.similarity_search_with_score(query, k=k)
  else:
    similar_docs = vectorstore.similarity_search(query, k=k)
  # print(similar_docs)
  return similar_docs 
  
# print(get_similiar_docs(query)[0])


## GPT model answer retrieval part
model_name = "gpt-3.5-turbo-16k"
# model_name = "gpt-3.5-turbo"
llm = ChatOpenAI(temperature=0, model_name=model_name)


prompt_template = """You are a helpful AI assistant that knows Bengali language. you are given information on \
          a famous poet named named kazi nazrul islam. you will be asked some questions on the information \
            \n\n\n

this is the context {context}
based on the context answer the following quesion and don't try to make up answers.\
 
Question: {question}
Answer in Bengali:"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)

def get_answer(query):
  similar_docs = get_similiar_docs(query, k = 8)
  # answer = chain.run(input_documents=similar_docs, question=query)

  answer = chain({"input_documents": similar_docs, "question": query},return_only_outputs=True)
  return answer



app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/reset/")
async def reset_vectorstores():
  pinecone.delete_index(index_name)
  createIndex(index_name)
  return "reseted vectorstore"



@app.get("/reload/")
async def load_vectorstores():
  docs = newDataLoad(directory = directory)
  addToIndex(index_name, documents=docs)
  return "re-filled the vectorstores from scratch"


class Question(BaseModel):
  query: str
  description: str | None = None


@app.post("/query/")
async def query_ans(question: Question):
  answer = get_answer(query=f'{question.query}')
  return answer


class Documents(BaseModel):
  index_name: str
  docs: list
  description: str | None = None

@app.post("/contribute/")
async def contribute_docs(documents: Documents):
  addToIndex(index_name=documents.index_name, documents=documents.docs)
  print("Added to the list")
  return documents.docs[0]


