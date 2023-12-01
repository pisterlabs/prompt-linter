import os
import dotenv
import weaviate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import weaviate
from langchain.vectorstores import Weaviate
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import Cohere

dotenv.load_dotenv()

weaviate_url = os.getenv("WEAVIATE_URL")
weaviate_key = os.getenv("WEAVIATE_KEY")
COHERE_APIKEY = os.getenv("COHERE_APIKEY")

loader = PyPDFLoader("pdfs/transformers.pdf")
pages = loader.load_and_split()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = splitter.split_documents(pages)

auth_config = weaviate.AuthApiKey(api_key=weaviate_key)

client = weaviate.Client(
    url=weaviate_url,
    additional_headers={"X-Cohere-Api-Key": COHERE_APIKEY},
    auth_client_secret=auth_config,
    startup_period=10)
    
schema = {
  "classes": [
    {
      "class": "Chatbot",
      "description": "A class for Chatbot",
      "vectorizer": "text2vec-cohere",
      "moduleConfig": {
        "text2vec-cohere": {
          "model": "embed-multilingual-v3.0",
          "type": "text"
          }
      },
      "properties": [
        {
            "dataType": ["text"],
            "description": "The text of the document",
            "moduleConfig": {
                "text2vec-cohere": {
                    "skip": False,
                    "vectorizerPropertyName": False,
                }
            },
            "name": "content",
        }
      ]
    }
  ]
}
# client.schema.create(schema)
vectorstore = Weaviate(
    client, "Chatbot", "content", attributes=["source"]
)
# print(client.schema.get())
text_meta_pair = [(doc.page_content, doc.metadata) for doc in docs]
texts, meta = list(zip(*text_meta_pair))
vectorstore.add_texts(texts, meta)

def chat(query):
    docs = vectorstore.similarity_search(query, k=4)
    chain = load_qa_chain(Cohere(cohere_api_key=COHERE_APIKEY))
    print(chain.run(input_documents=docs, question=query))

chat("What is the transformer architecture?")
