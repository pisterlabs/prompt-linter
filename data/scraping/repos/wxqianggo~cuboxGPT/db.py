from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
import os
from rich.console import Console
console = Console()
# @todo: add_documents,delte_collection


def initDatabase():
    docs_array = []
    directory = "text"
    persist_directory = 'db'
    # delete db folder
    if os.path.exists(persist_directory):
        os.system(f"rm -rf {persist_directory}")
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            file_path = os.path.join(directory, filename)
            loader = TextLoader(file_path)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(
                chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_documents(documents)
            for t in texts:
                t.metadata = {'source': filename.split(
                    '.txt')[0], 'url': documents[0].page_content.split('.\n')[1]}
            # join docs_array and docs two arrays
            docs_array = docs_array + texts
    embedding = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=docs_array, embedding=embedding,
        persist_directory=persist_directory)
    vectordb.persist()
    # vectordb = None
    # vectordb = Chroma(persist_directory=persist_directory,
    # embedding_function=OpenAIEmbeddings())
    # inspect(vectordb, methods=True)


if __name__ == '__main__':
    initDatabase()
