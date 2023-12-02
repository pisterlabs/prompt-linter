from dotenv import load_dotenv

from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

if __name__ == '__main__':
    load_dotenv()
    print('cooking book documents...')
    loader = DirectoryLoader('./rust-book/src', glob="**/*.md", use_multithreading=True)
    docs = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_documents(docs)

    vectordb = Chroma.from_documents(
        docs,
        embedding=OpenAIEmbeddings(),
        persist_directory='./vectordb'
    )
    vectordb.persist()
    print('done!')
