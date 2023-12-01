import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.document_loaders import TextLoader
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import pinecone


def load():
    files = ["docus/economia.txt", "docus/ingenieria-civil.txt", "docus/ingenieria-electrica.txt",
             "docus/ingenieria-electronica.txt", "docus/ingenieria-industrial.txt", "docus/ingenieria-sistemas.txt"]
    embeddings = OpenAIEmbeddings()
    for f in files:
        print("Estoy trabajando en subir: ",f)
        loader = TextLoader(f, "utf8")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        # initialize pinecone
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
            environment=os.getenv("PINECONE_ENV"),  # next to api key in console
        )

        index_name = "sainapsis"

        docsearch = Pinecone.from_documents(docs, embeddings, index_name=index_name)
        docsearch = Pinecone.from_existing_index(index_name, embeddings)


def search():
    pinecone.init(
        api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
        environment=os.getenv("PINECONE_ENV"),  # next to api key in console
    )
    embeddings = OpenAIEmbeddings()
    index_name = "sainapsis"
    docsearch = Pinecone.from_existing_index(index_name, embeddings)
    qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="refine", retriever=docsearch.as_retriever(type="similarity"))
    query = "Ingenieria de software esta acreditada?"
    print(qa.run(query))


if __name__ == '__main__':
    search()
