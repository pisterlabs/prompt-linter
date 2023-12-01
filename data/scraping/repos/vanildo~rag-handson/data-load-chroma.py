from dotenv import load_dotenv
import chromadb
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import loggingService
import os
from sentence_transformers import SentenceTransformer
import uuid

load_dotenv()
logger = loggingService.get_logger()

apikey = os.getenv("GEANAI_KEY", None)
class_name = os.getenv("WEVIATE_CLASS", 'Livros')
path = os.getenv("DATA_PATH", 'data')
model_name = os.getenv("MODEL_NAME_EMBEDDING", 'sentence-transformers/gtr-t5-large')
# model = SentenceTransformer(model_name)

client = chromadb.PersistentClient(path="db/")
pages = []
pdf_loader = PyPDFDirectoryLoader(path)
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 20, separators=['\n', '\n \n', '\n \n \n' ])
documents = pdf_loader.load_and_split(text_splitter=text_splitter)
  
logger.info(len(documents))

# client.delete_collection('Livros')
# client.create_collection('Livros')
    
def pdf_text_splitter(pdf_text) -> str:
  retorno = {'content': '', 'source': '', 'page': 0}
  
  retorno['content'] = getattr(pdf_text, 'page_content')
  retorno['souce'] = getattr(pdf_text, 'metadata')['source']
  retorno['page'] = getattr(pdf_text, 'metadata')['page']
  
  return retorno

def get_embedding(sentence: str,):
  model = SentenceTransformer(model_name)
  embeddings = model.encode(sentence)
  
  return embeddings

logger.debug(get_embedding('onde bras cubas morava'))

def load_documents():
  for doc in documents:
    logger.debug(pdf_text_splitter(doc))
    pages.append(pdf_text_splitter(doc))
    
  logger.info(len(pages))

def populate_db():
  load_documents()
  # collection = client.create_collection(name="Livros", embedding_function=get_embedding)
  # client.delete_collection('Livros')
  # collection = client.get_or_create_collection('Livros')
  collection = client.get_collection('Livros')
  
  documents_to_index = []
  embeddings = []
  metadatas = []
  ids = [f'{uuid.uuid4()}' for i in range(len(pages))]
  i = 0
  
  for document in pages:
    logger.info(f"importing question: {i+1}")
    i = i+1
    
    metadata = {
      "content": document["content"],
      "page": str(document["page"]),
      "source": document["source"],
    }
    vector = get_embedding(metadata.get('content'))
      
    documents_to_index.append(metadata.get('content'))
    embeddings.append(vector)
    metadatas.append(metadata)
    
  collection.add(
    documents=documents_to_index,
    embeddings=embeddings,
    metadatas=metadata,
    ids=ids,
  )

if __name__ == '__main__':
  # populate_db()
  collection = client.get_collection('Livros')
  query_embeddings = get_embedding("por que arthur dent foi despejado?").tolist()
  print(type(query_embeddings))
  
  results = collection.query(
    query_embeddings=query_embeddings,
    n_results=4
  )

  print(results)
