from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

def get_vectorstore(documents):
    embeddings = OpenAIEmbeddings()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    )
    #text_splitter = CharacterTextSplitter(
    #    chunk_size=1000, chunk_overlap=0
    #)
    docs = text_splitter.split_documents(documents=documents)
    if(len(docs) == 0):
        return False
    else:
        for idx, doc in enumerate(docs, start=1):
            doc.metadata['doc_id'] = idx
            doc.metadata['source'] = doc.metadata['source']#.split("\\")[-1]
        vectorstore = FAISS.from_documents(documents=docs, embedding=embeddings)
       
        return vectorstore

