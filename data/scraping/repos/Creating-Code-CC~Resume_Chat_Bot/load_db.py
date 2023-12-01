import sys
import lib
import openai
import os
import datetime
from dotenv import load_dotenv, find_dotenv

llm_name=""
current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"


_=load_dotenv(find_dotenv())
openai.api_key=os.environ['OPENAI_API_KEY']
# def func(file, kwargs={})
def load_db(file, chain_type, k):
# Load Documents
    loader = lib.PyPDFLoader(file)
    documents = loader.load() # Original documents
    # Define Text Splitter
    text_splitter = lib.RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=150)
    # Split documents
    docs = text_splitter.split_documents(documents) # New Split Documents

    # Create embeddings before creating vector store
    embeddings = lib.OpenAIEmbeddings()
    # Create Vector Store with embeddings for Semantic Search
    db = lib.DocArrayInMemorySearch.from_documents(documents = docs,embedding=embeddings)

    # Create retriever
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k":k}) # Have ChatGPT explain this line

    qa = lib.ConversationalRetrievalChain.from_llm(llm=lib.ChatOpenAI(model_name=llm_name, temperature=0), chain_type=chain_type, retriever=retriever, return_source_documents= True, return_generated_question=True,)

    return qa
   
