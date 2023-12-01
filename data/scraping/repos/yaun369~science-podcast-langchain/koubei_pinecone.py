import os
from dotenv import load_dotenv
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import Pinecone
import pinecone

load_dotenv()
OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
OPENAI_API_BASE = os.environ['OPENAI_API_BASE']

PINECONE_API_KEY = os.environ['PINECONE_API_KEY']
PINECONE_ENV = os.environ['PINECONE_ENV']

PINECONE_INDEX = 'carkoubei'

pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_ENV  # next to api key in console
)

embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY, openai_api_version='2020-11-07')

if PINECONE_INDEX not in pinecone.list_indexes():
    loader = CSVLoader(file_path='koubei.csv')
    data = loader.load()
    docs = data[:10]
    print(len(docs))
    Pinecone.from_documents(docs, embeddings, index_name=PINECONE_INDEX)

docsearch = Pinecone.from_existing_index(
    index_name=PINECONE_INDEX, embedding=embeddings)

query = 'CR-V值得买么？'
search_docs = docsearch.similarity_search(query, k=3)
# print(search_docs)

llm = ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')
chain = load_qa_chain(llm, chain_type='stuff')
results = chain.run(input_documents=search_docs, question=query)
print(f'Q: {query}')
print(f'A: {results}')
