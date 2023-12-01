"""
In this module, files are uploaded, chunked and embedded using openAI
embeddings. The langchain framework for working on large language model Apps
is employed to handle the breakdown of texts in the pdf file into smaller chunks,
after which they are split, embedded and stored in a vectore store database.
Chroma is the vectordb used in the project
"""
import configAi
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader

def load_db(file, chain_type, k):
    """process pdf files for query"""
    # load documents
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    docs = text_splitter.split_documents(documents)
    # define embedding
    embeddings = OpenAIEmbeddings()
    # create vector database from data
    persist_directory = 'docs/chroma'
    #rm -rf ./docs/chroma  # remove old database files if any
    vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=persist_directory,
            )
    vectordb.persist()
    #db = DocArrayInMemorySearch.from_documents(docs, embeddings)
    # define retriever
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k": k})
    # create a chatbot chain. Memory is managed externally.
    qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0), 
        chain_type=chain_type, 
        retriever=retriever,
        return_source_documents=True,
        return_generated_question=True,
    )
    return qa 
