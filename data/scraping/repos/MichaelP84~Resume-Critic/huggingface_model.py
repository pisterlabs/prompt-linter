import os
import requests
from langchain.document_loaders import TextLoader
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from dotenv import load_dotenv


pdfs = []
txts = []
load_dotenv()
os.environ["HUGGINGFACEHUB_API_TOKEN"] = os.getenv('HUGGINGFACEHUB_API_TOKEN')


def find_files(source="./resources"):
    for file in os.listdir(source):
        file = os.path.join(source, file)
        if file.endswith(".pdf"):
            pdfs.append(file)
        elif file.endswith(".txt"):
            txts.append(file)


def load_files():
    pdf_loader = UnstructuredPDFLoader(pdfs[0])
    # txt_loaders = TextLoader(txts)

    # print(txt_loaders)
    loaded_docs = pdf_loader.load()
    return loaded_docs

def splitDocument(loaded_docs):
    # Splitting documents into chunks
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=10)
    chunked_docs = splitter.split_documents(loaded_docs)
    return chunked_docs

def createEmbeddings(chunked_docs):
    # Create embeddings and store them in a FAISS vector store
    embedder = HuggingFaceEmbeddings()
    vector_store = FAISS.from_documents(chunked_docs, embedder)
    return vector_store

def loadLLMModel():
    llm=HuggingFaceHub(repo_id="declare-lab/flan-alpaca-large", model_kwargs={"temperature":0, "max_length":512})
    chain = load_qa_chain(llm, chain_type="stuff")
    return chain

def askQuestions(vector_store, chain, question):
    # Ask a question using the QA chain
    similar_docs = vector_store.similarity_search(question)
    response = chain.run(input_documents=similar_docs, question=question)
    return response

if '__main__' == __name__:
    chain = loadLLMModel()

    find_files()
    LOCAL_loaded_docs = load_files()
    LOCAL_chunked_docs = splitDocument(LOCAL_loaded_docs)
    LOCAL_vector_store = createEmbeddings(LOCAL_chunked_docs)
    resume_section = "Texas Guadaloop Hyperloop Research & Engineering – Data Analyst, Integrating and maintaining all of the business team and engineering teams’ data onto AWS S3 buckets in order to streamline and organize data storage, analysis, and export. Building a website for the team to display team achievements, important data, and student culture"
    LOCAL_response = askQuestions(LOCAL_vector_store, chain, f"Using this resume as reference for writing style and direction, revise the following resume bullet point: {resume_section}")
    print(LOCAL_response)   