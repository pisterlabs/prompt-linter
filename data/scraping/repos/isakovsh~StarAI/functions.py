from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import Qdrant
from langchain.embeddings.openai import OpenAIEmbeddings
import qdrant_client
import os
import openai
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader 
from langchain.text_splitter import CharacterTextSplitter
import re

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

client = qdrant_client.QdrantClient(
    os.getenv("QDRANT_HOST"),
    api_key = os.getenv("QDRANT_API_KEY")
)

embeddings = OpenAIEmbeddings()

vectorstore = Qdrant(
        client=client,
        collection_name=os.getenv("QDRANT_COLLECTION_NAME"),
        embeddings=embeddings
    )


def add_pdf_to_vector_stor(file):
    
  # get text from pdf file
  loader = PyPDFLoader(file)
  docs = loader.load()
  text = " ".join(docs[i].page_content for i in range(len(docs)))
  text = re.sub('\n','',text)

  # create chunks
  splitter = CharacterTextSplitter(
       separator='\n',
       chunk_size =1000,
       chunk_overlap = 200,
       length_function = len
    )
  text = splitter.split_text(text)

  # add text chunks to vector store
  result = vectorstore.add_texts(text)

  return result


def qa(query):
    llm = OpenAI()
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
        ) 
    result = qa.run(query)
    return result
    