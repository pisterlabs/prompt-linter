from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.document_loaders import TextLoader
from langchain.document_loaders import TextLoader

def load_docs():
    loader = TextLoader('state_of_the_union.txt')
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()

    qdrant = Qdrant.from_documents(docs, embeddings, host='localhost', port=6333)
    query = "What did the president say about Ketanji Brown Jackson"
    docs = qdrant.similarity_search(query)
    print(docs[0])

    


