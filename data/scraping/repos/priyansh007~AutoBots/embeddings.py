from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
import pickle
from langchain import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import CohereEmbeddings
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.docstore.document import Document
import re
import codecs
from bs4 import BeautifulSoup
import PyPDF2

#For Loading The documents
####################################### New Version
# Create text splitter by paragraphs
# by empty lines
def paragraph_text_splitter(text, source):
    paragraphs = re.split(r'\.', text)
    paragraphs = [' '.join(paragraphs[i:i+5]) for i in range(0, len(paragraphs), 5)]
    paragraphs = [Document(page_content=par, metadata={"source": source}) for par in paragraphs]
    return paragraphs

# For Loading The documents
def doc_load(files):
    documents = []

    for file in files:
        ext = os.path.splitext(file.name)[1]
        
        if ext.lower() == '.md':
            documents.append(Document(page_content=str(file.read(), encoding='utf-8', errors='ignore'), metadata={"source": file.name}))
        
        elif ext.lower() == '.html':
            soup = BeautifulSoup(file, 'html.parser')
            paragraphs = [p.get_text() for p in soup.find_all('p')]
            text = '\n\n'.join(paragraphs)
            documents.append(Document(page_content=text, metadata={"source": file.name}))

        elif ext.lower() == '.pdf':
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            documents.append(Document(page_content=text, metadata={"source": file.name}))

    # use the text splitter by paragraphs
    splitted = []
    for doc in documents:
        paragraphs = paragraph_text_splitter(doc.page_content, doc.metadata['source'])
        splitted.extend(paragraphs)

    return splitted
##########################################


# #For Loading The documents
# def doc_load(files):
#   documents = []

#   for file in files:
#       ext = os.path.splitext(file.name)[1]
#       if ext.lower() in ['.md', '.txt']:
#           documents.append(Document(page_content=str(file.read(), encoding='utf-8', errors='ignore'), metadata={"source": file.name}))
#   text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=80)
#   splitted = text_splitter.split_documents(documents)
#   print(splitted)
#   return splitted

#We shall not call embedding function again and again, Instead we shall save our embedding in some pickle file locally
def save_embedding_into_pickle(file_path):
  sentence_embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
  with open(file_path, 'wb') as f:
      pickle.dump(sentence_embedding, f)
  print('done with saving embeddings into pickle')

def load_embedding_from_pickle(path):
  with open(path, 'rb') as f:
    embedding = pickle.load(f)
    return embedding


#Once User uploads all documents we shall also save chroma using this function
def save_chroma_using_embedding(documents,
                                embedding_pickle_path,
                                vector_path):
  embedding = load_embedding_from_pickle(embedding_pickle_path)
  db = Chroma.from_documents(documents, embedding, persist_directory=vector_path)
  db.persist()

# Once User ask questions we only need to load following function

def load_chroma_with_query_without_compressor(vector_path,
                           embedding_pickle_path,
                           query,
                           num_of_facts=10):
  embedding = load_embedding_from_pickle(embedding_pickle_path)
  db = Chroma(persist_directory=vector_path, embedding_function=embedding)
  db.get()
  retriever = db.as_retriever(search_kwargs={"k": num_of_facts})
  ret_ans = retriever.get_relevant_documents(query)
  unique_docs = [doc for i, doc in enumerate(ret_ans) if doc not in ret_ans[:i]]
  return unique_docs


def load_chroma_with_query_with_compressor(vector_path,
                           embedding_pickle_path,
                           query,
                           api_key,
                           num_of_facts=10):
  os.environ["COHERE_API_KEY"]= api_key
  embedding = load_embedding_from_pickle(embedding_pickle_path)
  db = Chroma(persist_directory=vector_path, embedding_function=embedding)
  compressor = CohereRerank()
  db.get()
  retriever = db.as_retriever(search_kwargs={"k": num_of_facts})
  compression_retriever = ContextualCompressionRetriever(
      base_compressor=compressor,
      base_retriever=retriever
  )
  ret_ans = compression_retriever.get_relevant_documents(query)
  unique_docs = [doc for i, doc in enumerate(ret_ans) if doc not in ret_ans[:i]]
  return unique_docs