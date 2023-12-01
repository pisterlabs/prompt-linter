import os
import pinecone
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain


# load the environment variables
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENVIRON = os.environ.get('PINECONE_ENVIRONMENT')

model_name = 'gpt-3.5-turbo'
llm = ChatOpenAI(model=model_name)

directory = './web_dev/AI/data'

# load the documents from directory


def load_documents(directory, filename):
    loader = DirectoryLoader(directory, filename)
    documents = loader.load()
    return documents

# split files into chunks


def split_documents(documents, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)
    return docs


# load the embeddings
model_name = 'text-embedding-ada-002'
embeddings = OpenAIEmbeddings(model=model_name)

# load the vector store
pinecone.init(
    PINECONE_API_KEY=PINECONE_API_KEY,
    PINECONE_ENVIRONMENT=PINECONE_ENVIRON
)

# save the embeddings to the vector store
index_name = 'demo-index'


def save_embeddings(docs, embeddings, index_name=index_name):
    if Pinecone.from_documents(docs, embeddings, index_name=index_name):
        return True


# load existing index
index = Pinecone.from_existing_index(index_name, embeddings)

# querry the index store


def query_index(query, k=3):
    search_results = index.similarity_search(query, k=k)
    return search_results


def get_answer(query):
    chain = load_qa_chain(llm, chain_type="stuff")
    similar_docs = query_index(query)
    answer = chain.run(input_documents=similar_docs, question=query)
    return answer
