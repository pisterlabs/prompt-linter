from langchain.document_loaders.csv_loader import CSVLoader
import os
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_API_ENV = os.getenv('PINECONE_API_ENV')

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

products = 'SampleProducts.csv'

pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "store-data" # put in the name of your pinecone index here

data = CSVLoader(file_path=products, encoding="utf-8", csv_args={
        'delimiter': ','}).load()

print(len(data))

docsearch = Pinecone.from_texts([t.page_content for t in data], embeddings, index_name=index_name)