from langchain.document_loaders import WebBaseLoader, TextLoader 
from main import hf
import torch
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_documents(documents):
    loader = TextLoader(documents)
    documents = loader.load()

    return documents

def load_web_documents(web_links):
    loader = WebBaseLoader(web_links)
    documents = loader.load()

    return documents

def get_web_answers(sources, mode, question):
    documents = []

    if mode == "web":
        documents = load_web_documents(sources)
    else:
        documents = load_documents(sources)

    from langchain.text_splitter import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    all_splits = text_splitter.split_documents(documents)

    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.vectorstores import FAISS

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": device}

    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    # storing embeddings in the vector store
    vectorstore = FAISS.from_documents(all_splits, embeddings)

    from langchain.chains import ConversationalRetrievalChain

    chain = ConversationalRetrievalChain.from_llm(hf, vectorstore.as_retriever(), return_source_documents=True)

    chat_history = []

    if sys.argv[1] != None:
        query = sys.argv[1]
    elif question != None:
        query = question
    else:
        query = "Tell me about the modules in Braggi EMS."

    # print("Question: ", question)

    result = chain({"question": query, "chat_history": chat_history})

    # print("Answer: ", result['answer'])
    # print("\n\n ============================ \n\n")
    # print("Source: ", result['source_documents'])
    
    return result
