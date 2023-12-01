from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os


def load_documents(pdf_paths):
    documents = []
    for pdf_path in pdf_paths:
        loader = PyMuPDFLoader(pdf_path)
        documents.extend(loader.load())
    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len, separators="\n"
    )
    return text_splitter.split_documents(documents)


def create_vectordb(
    pdf_paths, persist_directory, pdf_video_path, video_persist_directory
):
    documents = load_documents(pdf_paths)
    texts = split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(
        documents=texts, embedding=embeddings, persist_directory=persist_directory
    )
    vectordb.persist()

    retriever = vectordb.as_retriever(search_kwargs={"k": 6})
    llm = ChatOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        model_name="gpt-3.5-turbo-0301",
        temperature=0,
        max_tokens=500,
        verbose=True,
    )

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm, chain_type="stuff", retriever=retriever, verbose=True
    )

    video_documents = load_documents(pdf_video_path)
    video_texts = split_documents(video_documents)
    video_embeddings = OpenAIEmbeddings()
    video_vectordb = Chroma.from_documents(
        documents=video_texts,
        embedding=video_embeddings,
        persist_directory=video_persist_directory,
    )
    video_vectordb.persist()
    # video_retriever = video_vectordb.as_retriever(search_kwargs={"k": 6})

    return qa, video_vectordb
