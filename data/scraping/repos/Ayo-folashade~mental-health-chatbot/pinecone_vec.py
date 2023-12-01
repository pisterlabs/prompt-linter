import openai
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.llms import OpenAI
from langchain.vectorstores import Pinecone
import pinecone
import os
from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

embeddings = OpenAIEmbeddings()

file = 'data.csv'
loader = CSVLoader(file_path=file)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

# initialize pinecone
pinecone.init(
    api_key=os.environ['PINECONE_API_KEY'],  # find at app.pinecone.io
    environment=os.environ['PINECONE_ENV']  # next to api key in console
)

os.environ["PINECONE_INDEX_NAME"] = "demo-1536"

docsearch = Pinecone.from_documents(docs, embeddings, index_name=os.environ['PINECONE_INDEX_NAME'])

docsearch = Pinecone.from_existing_index(os.environ['PINECONE_INDEX_NAME'], embeddings)

query = "i lost someone"
docs = docsearch.similarity_search(query)

model = OpenAI(model_name="text-davinci-003")
sources_chain = load_qa_with_sources_chain(model, chain_type="refine")
result = sources_chain.run(input_documents=docs, question=query)
print(result)
