from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticKnnSearch, Pinecone, Weaviate, FAISS
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredHTMLLoader
import os
import pinecone
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
import time

from keys import OPENAI_API_KEY

# PINECONE_API_KEY = PINECONE_API_KEY
# PINECONE_ENV = PINECONE_ENV

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def pinecone_create_vector_store(software):
    print(software)
    loader = DirectoryLoader(f'/home/yuvraj/projects/docai/test/{software}', glob="**/*.html", use_multithreading=True, loader_cls=UnstructuredHTMLLoader, show_progress=True)
    data = loader.load()
    print(data)

    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    docs = text_splitter.split_documents(data)

    # file = open('docs.txt', 'w')
    # for item in docs:
    #     file.write(item.page_content+"\n")
    # file.close()

    embeddings = OpenAIEmbeddings()
    # pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)

    # index_name = software
    # pinecone.create_index(name=index_name, metric="cosine", shards=1, dimension=16000)

    print(len(docs))
    vector_store = None
    token_count = 0
    pre_tokens = 0
    with get_openai_callback() as cb:
        for i, doc in enumerate(docs):
            print(i)
            if vector_store is None:
                vector_store = FAISS.from_documents([doc], embeddings)
            else:
                vector_store.add_documents([doc])

            print(f'added to vector store doc {i}')

            print(cb.total_tokens)
            print(pre_tokens)
            token_count += cb.total_tokens - pre_tokens
            pre_tokens = cb.total_tokens
            print(token_count)
            
            if token_count > 995000:
                time.sleep(45)
                token_count = 0

            print(token_count)
                
    
    vector_store.save_local(f'/home/yuvraj/projects/docai/vector_stores/{software}', index_name=software)

    return(f'uploaded vector {software}')

    # vector = FAISS.from_documents(docs, embeddings, index_name=index_name)

    # vector.save_local()
