# Ejemplo de uso de la librería langchain para crear un chatbot de recuperación de información conectándose a gDrive

from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import GoogleDriveLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from colorama import Fore

# Setear el ID de la carpeta de Google Drive en folder_id
folder_id = "1GKpCcYQ_ZgYeE3SploS50gOwdIzPlKq_"  # LLM Ejemplo

loader = GoogleDriveLoader(folder_id=folder_id, recursive=False)
docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000, chunk_overlap=0, separators=[" ", ",", "\n"]
)

texts = text_splitter.split_documents(docs)
embeddings = OpenAIEmbeddings()
vectordb = Chroma.from_documents(
    documents=docs, embedding=embeddings, persist_directory=".chroma/vectordb"
)
retriever = vectordb.as_retriever()

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

while True:
    print(Fore.WHITE)
    query = input("> ")
    answer = qa.run(query)

    # docs = vectordb.similarity_search(query)

    print(Fore.GREEN, answer)

    # vectordb.persist()
    # vectordb = None
