import os
from langchain.document_loaders import PyPDFLoader 
from langchain.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma

class Persistence:
    dir = "embeddings"
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 0)
    all_splits = []


    def get_storage(inputs):
        if True or os.path.exists(inputs + "/*.docx") or os.path.exists(inputs + "/*.pdf"):
            print('Re-indexing')
            return Persistence.build_index(inputs, Persistence.dir)
        else:
            print('Accessing existing index')
            return Persistence.reload_index(Persistence.dir)

    def build_index(source_path, index_path):
        for filename in os.listdir(source_path):
                if filename.endswith(".pdf"):
                    print('Indexing PDF ' + filename)
                    loader = PyPDFLoader(source_path + '/' + filename)
                elif filename.endswith(".doc") or filename.endswith(".docx"):
                    print('Indexing Word document ' + filename)
                    loader = Docx2txtLoader(source_path + '/' + filename)

                data = loader.load()
                Persistence.all_splits += Persistence.text_splitter.split_documents(data)

        return Chroma.from_documents(documents=Persistence.all_splits, embedding=OpenAIEmbeddings(), persist_directory=index_path)

    def reload_index(index_path):
         return Chroma(persist_directory="embeddings", embedding_function=OpenAIEmbeddings())
