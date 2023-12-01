import os
import openai
import pinecone
import gradio as gr
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

openai.api_key = os.environ["OPENAI_API_KEY"] 

pinecone.init(api_key=os.environ["PINECONE_API_KEY"],
              environment=os.environ["PINECONE_ENV"])
              
def load_docs(directory):
  loader = DirectoryLoader(directory)
  return loader.load()
  
def split_docs(documents):
  splitter = RecursiveCharacterTextSplitter()
  return splitter.split_documents(documents)
  
def index_docs(docs):
  embeddings = OpenAIEmbeddings()
  index = Pinecone.from_documents(docs, embeddings, index_name = "resume")
  return index

def get_similar_docs(query, index):
  return index.similarity_search(query)
  
def answer_question(query, docs, llm):
  chain = load_qa_chain(llm)
  return chain.run(input_documents=docs, question=query)
  
def qa_pipeline(query, docs_dir):
  documents = load_docs(docs_dir)
  docs = split_docs(documents)
  index = index_docs(docs)
  similar_docs = get_similar_docs(query, index)
  llm = OpenAI(model_name="text-curie-001")
  return answer_question(query, similar_docs, llm)

iface = gr.Interface(fn=qa_pipeline, 
                     inputs=["text", "text"], 
                     outputs="text")

iface.launch()