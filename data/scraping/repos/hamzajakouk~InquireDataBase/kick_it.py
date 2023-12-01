from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os
load_dotenv()

llm = GooglePalm(google_api_key=os.environ['API_KEY'], temprature=0.6)
vectordb_filename = "faiss_db"
embeddings = HuggingFaceInstructEmbeddings()
def create_vectore():
    loader = CSVLoader(file_path="codebasics_faqs.csv", source_column="prompt", encoding='ISO-8859-1')
    data = loader.load()
    vectordb = FAISS.from_documents(embedding = embeddings, documents = data)
    vectordb.save_local(vectordb_filename)

def get_question():
    vectordb = FAISS.load_local(vectordb_filename, embeddings)
    retriever = vectordb.as_retriever()
    prompt_template = """i want you to respect the context try to provie as much as possible from the "response" section don't try to make thing from your own if the answer is not exist just jay basically "i don't know" kindly
    CONTEXT : {context}
    QUESTION: {question}
    """
    PROMPT = PromptTemplate(template = prompt_template, input_variables=["context" , "question"])
    chain = RetrievalQA.from_chain_type(llm = llm , chain_type = "stuff", retriever=retriever ,input_key = "query" , return_source_documents=True, chain_type_kwargs={"prompt":PROMPT} )
    return chain
