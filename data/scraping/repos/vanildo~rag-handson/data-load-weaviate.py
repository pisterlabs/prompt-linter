import unicodedata
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import loggingService
import os
import weaviate

load_dotenv()
logger = loggingService.get_logger()

class_name = os.getenv("WEVIATE_CLASS", 'Livros')
path = os.getenv("DATA_PATH", 'data')
weaviate_url = os.getenv("WEAVIATE_URL", 'http://127.0.0.1:8080')

client = weaviate.Client(url=weaviate_url,)

pages = []
pdf_loader = PyPDFDirectoryLoader(path)
text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 20, separators=['\n', '\n \n', '\n \n \n' ])
documents = pdf_loader.load_and_split(text_splitter=text_splitter)

class_document = {
      "class": "Livros",
      "description": "Dados do documento PDF",
      "moduleConfig": {
        "text2vec-transformers": {
          "poolingStrategy": "masked_mean",
          "vectorizeClassName": False
        }
      },
      "properties": [
        {
          "dataType": [
            "text"
          ],
          "description": "Conteúdo do documento PDF",
          "moduleConfig": {
            "text2vec-transformers": {
              "skip": False,
              "vectorizePropertyName": False
            }
          },
          "name": "content"
        },
        {
          "dataType": [
            "text"
          ],
          "description": "",
          "moduleConfig": {
            "text2vec-transformers": {
              "skip": False,
              "vectorizePropertyName": False
            }
          },
          "name": "source"
        },
        {
          "dataType": [
            "text"
          ],
          "description": "",
          "moduleConfig": {
            "text2vec-transformers": {
              "skip": False,
              "vectorizePropertyName": False
            }
          },
          "name": "page"
        }
      ],
      "vectorizer": "text2vec-transformers"
      # "vectorizer": "none"
    }
  
print(len(documents))
# logger.info(len(documents))
    
def pdf_text_splitter(pdf_text) -> str:
  retorno = {'content': '', 'source': '', 'page': 0}
  
  retorno['content'] = getattr(pdf_text, 'page_content')
  retorno['souce'] = getattr(pdf_text, 'metadata')['source']
  retorno['page'] = getattr(pdf_text, 'metadata')['page']
  
  return retorno

def load_documents():
  for doc in documents:
    logger.debug(pdf_text_splitter(doc))
    pages.append(pdf_text_splitter(doc))
    
  logger.info(len(pages))

def populate_db():
  client.schema.delete_class('Livros')
  client.schema.create_class(class_document)
  load_documents()
  
  client.batch.configure(batch_size=5)  # Configure batch
  with client.batch as batch:
    i = 0
    batch.configure(batch_size=5)  # Configure batch
    
    for page in pages:
      logger.info(f"importing question: {i+1}")
      i = i+1
      
      properties = {
        "content": unicodedata.normalize("NFKD", page["content"]),
        "page": str(page["page"]),
        "source": page["source"],
      }
      
      batch.add_data_object(properties, 'Livros')

if __name__ == '__main__':
  # populate_db()
  
  result = (client.query
    .get('Livros', ["content", "source", "page"])
    .with_additional(["certainty", "distance"]) # note that certainty is only supported if distance==cosine
    # .with_near_vector({
      # "vector": vector,
      # "certainty": 0.8
    # })
    .with_near_text({'concepts': 'por que arthur dent foi despejado?', 'certainty': 0.6})
    .with_limit(4)
    .do()
  )
  class_name = 'Livros'
  print(result)
  print(result['data']['Get'][class_name])
