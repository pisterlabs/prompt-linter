
# [1]
import sys
# import document loader from lamgcahin
from langchain.document_loaders import  OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma,Pinecone
from langchain import VectorDBQA
from langchain.llms import Cohere,OpenAI
from langchain.embeddings import CohereEmbeddings
import pinecone
import os
from langchain.chains.question_answering import load_qa_chain

# defining the cohere keys
COHERE_API_KEY = 'lgi7A2ZBRIswmmUy3FIB0AbjfNhEnvWtgEXnElPi'
EMBEDDING_TYPE = 'cohere'
PINECONE_KEY = os.getenv("PINECONE_KEY")
#defining the loader
def load_data(data_path='https://django.readthedocs.io/_/downloads/en/latest/pdf/',loader_type='online'
              ):

    if loader_type == 'online':
        loader = OnlinePDFLoader('https://django.readthedocs.io/_/downloads/en/latest/pdf/')
    data = loader.load()
    print(f"--- No of pages: {len(data)} \n")

    # Chunking up the data
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = splitter.split_documents(data)
    print(f"--- No of chunks: {len(texts)}")
    return texts


def generate_embeddings(texts, embedding_type=EMBEDDING_TYPE):
    if embedding_type == 'cohere':
        embeddings = CohereEmbeddings(cohere_api_key=COHERE_API_KEY)
    elif embedding_type == 'openai':
        embeddings = OpenAIEmbeddings(openai_api_key='OPENAI_API_KEY')
    return embeddings

def initialize_vecstore(embeddings,texts, vector_store='pinecone'):
    if vector_store == 'pinecone':
        #initialize pinecone
        # pinecone.init( api_key=PINECONE_KEY,environment='us-west1-gcp')
        index_name = 'testindex-co'

        # if index_name not in pinecone.list_indexes():
        #     if EMBEDDING_TYPE == 'cohere':
        #         pinecone.create_index(index_name, vector_size=4096, metric='cosine')
        #     elif EMBEDDING_TYPE == 'openai':
        #         pinecone.create_index(index_name, vector_size=768, metric='cosine')
        # else:
        #     index_pine = pinecone.Index("cohere")

        search_docs = Pinecone.from_texts([t.page_content for t in texts],embeddings, index_name=index_name)

    return search_docs

def initialize_llm(llmtype='cohere'):
    if llmtype == 'cohere':
        llm = Cohere(cohere_api_key=COHERE_API_KEY)
    elif llmtype == 'openai':
        llm = OpenAI(openai_api_key='OPENAI_API_KEY')
    return llm



def query_vecstore(search_docs, query,llm):
    topk_docs = search_docs.similarity_search(query, include_metadata=True)

    llm = initialize_llm()
    # qa = VectorDBQA.from_chain_type(llm=llm, chain_type="stuff", vectorstore=search_docs)
    chain = load_qa_chain(llm, chain_type="stuff")
    
    output = chain.run(query, topk_docs)
    print(output)
    return output
    


def run_process():
    texts = load_data()
    embeddings = generate_embeddings(texts)
    search_docs = initialize_vecstore(embeddings,texts)

    llm = initialize_llm()
    query = 'How to setup models in django?'
    output = query_vecstore(search_docs, query,llm)
    return output
# running the process
run_process()



'''
search_docs = Pinecone.from_texts([t.page_content for t in texts],cohere_embeddings, index_name=index_name)
q = input("Enter your query: ")
print(search_docs.similarity_search(q,include_metadata=True))

# docsearch = Chroma.from_documents(texts, embeddings)

from langchain.llms import Cohere
cohere = Cohere(model="gptd-instruct-tft", cohere_api_key="lgi7A2ZBRIswmmUy3FIB0AbjfNhEnvWtgEXnElPi")

qa = VectorDBQA.from_chain_type(llm=cohere, chain_type="stuff", vectorstore=search_docs)
'''