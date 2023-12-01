import pinecone      

import time

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.document_loaders import  TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
embed_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'device': 'cpu', 'batch_size': 32}
)


def data_loader():
    loader =  TextLoader('../test.txt')
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=400,
                                               chunk_overlap=20, separators=["\n\n", "\n", " ",""])
    texts = text_splitter.split_documents(documents)
    return texts

texts = data_loader()

pinecone.init(      
)      
index_name = 'llama2test'
if index_name not in pinecone.list_indexes():
    pinecone.create_index(name=index_name, dimension=384, metric='cosine'
        )
    
while not pinecone.describe_index(index_name).status['ready']:
    time.sleep(1)
    
index = pinecone.Index(index_name=index_name)
print(index.describe_index_stats())