from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS, Pinecone
from langchain.llms import HuggingFaceHub
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
import pinecone
import environ

env = environ.Env()
environ.Env.read_env()

pinecone.init(api_key=env("PINECONE_API_KEY"), environment=env("PINECONE_ENV"))


def get_pdf_text(pdf_docs):
    text = ""
    for key in pdf_docs:
        pdf_reader = PdfReader(pdf_docs[key])
        for page in pdf_reader.pages:
            text += page.extract_text()
    # print(text)
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    chunks = text_splitter.split_text(text)
    # print(chunks)
    return chunks


def get_vectorstore(text_chunks):
    # embedding = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    embeddings = OpenAIEmbeddings()
    # vectorstore = FAISS.from_texts(text_chunks, embeddings)
    vectorstore = Pinecone.from_texts(
        text_chunks, embeddings, index_name="langchain-demo"
    )
    return vectorstore


def get_conversation_chain(vector_store):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(
    #     repo_id="google/flan-t5-base",
    #     model_kwargs={"temperature": 0.5, "max_length": 512},
    # )
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    # conversation_chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm, retriever=vector_store.as_retriever(), memory=memory
    # )
    conversation_chain = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vector_store.as_retriever()
    )

    return conversation_chain
