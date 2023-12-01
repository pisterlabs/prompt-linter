

# -- standard imports
import os
from retry import retry 

# == langchain imports
# ---- llms 
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# ---- docs
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import YoutubeLoader

# ---- vectorstores
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

# ---- retrievers
from langchain.document_transformers import EmbeddingsRedundantFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline, EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever

# == local imports
from ._helpers import checkYoutubeLink


# ==================================================
# -- Loaders
# ==================================================

def loadLLM(temp = 0.6, model = ''): 
  OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') 

  if len(model) > 0: 
    model = 'gpt-3.5-turbo-16k' if model == 'chat' else model
    llm = ChatOpenAI(model_name=model, temperature=temp)

  else:
    llm = OpenAI(temperature=temp, openai_api_key=OPENAI_API_KEY)

  return llm

def generateDocs(text, chuck_size=1000, overlap=0, metadata = dict()):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chuck_size, chunk_overlap=overlap)
  texts = text_splitter.split_text(text)

  if len(metadata) > 0:
    docs = [Document(page_content=t, metadata=[metadata]) for t in texts]
  else:
    docs = [Document(page_content=t,) for t in texts]

  return docs

def loadYoutubeLoader(url):
  if not checkYoutubeLink(url):
    raise ValueError('Provided URL is not a valid youtube link.')
  
  loader = YoutubeLoader.from_youtube_url(url)
  transcript = loader.load()

  return transcript

def predict(prompt, temp = 0.6, model = 'chat'):
  llm = loadLLM(temp=temp, model=model)

  prediction = llm.predict(prompt)

  return prediction


# ==================================================
# -- Vector Stores
# ==================================================

def createVectorstore_api(docs, embeddings, persist_directory = ''): 
  # -- persist https://github.com/hwchase17/chroma-langchain/blob/master/persistent-qa.ipynb

  vectorstore = Chroma.from_documents(docs, embeddings)

  return vectorstore

def addTextToVectorStore_api(vectorstore, text, metadata = dict()): 

  # -- generate docs (split text into chunks)
  docs = generateDocs(text, metadata=metadata)

  # -- create new vectorstore from docs if none provided
  if type(vectorstore) != Chroma:
    embeddings = OpenAIEmbeddings()
    vectorstore = createVectorstore_api(docs, embeddings)
    return vectorstore
  
  # -- add docs to vectorstore
  texts = [doc.page_content for doc in docs]
  metadatas = [metadata for doc in docs] if len(metadata) == 0 else None

  vectorstore.add_texts(texts) if metadatas == None else vectorstore.add_texts(texts, metadatas=metadatas)

  return vectorstore

def retrieveDocs_api(vectorstore, query, k = 5):
  docs = vectorstore.similarity_search(query, k=k)

  return docs

def retrieveCompressedDocs_api(vectorstore, query, similarity_threshold = 0.76):
  embeddings = OpenAIEmbeddings()

  splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
  redundant_filter = EmbeddingsRedundantFilter(embeddings=embeddings)
  relevant_filter = EmbeddingsFilter(embeddings=embeddings, similarity_threshold= similarity_threshold)
  pipeline_compressor = DocumentCompressorPipeline(
      transformers=[splitter, redundant_filter, relevant_filter]
  )

  compression_retriever = ContextualCompressionRetriever(base_compressor=pipeline_compressor, base_retriever=vectorstore.as_retriever())

  compressed_docs = compression_retriever.get_relevant_documents(query)

  return compressed_docs