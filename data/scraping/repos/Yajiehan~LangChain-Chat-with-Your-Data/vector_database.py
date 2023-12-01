# Step 1: Overview of the workflow for RAG 
# Step 1.1: Set up environment
import os
import openai

from dotenv import load_dotenv, find_dotenv
# read local .env file
_ = load_dotenv(find_dotenv()) 

openai.api_key  = os.environ['OPENAI_API_KEY']

import datetime
current_date = datetime.datetime.now().date()
if current_date < datetime.date(2023, 9, 2):
    llm_name = "gpt-3.5-turbo-0301"
else:
    llm_name = "gpt-3.5-turbo"
print(llm_name)

# Step 2: Load document and create VectorDB
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
persist_directory = 'docs/chroma/'
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding)

# Step 3: Similarity Search to select relevant chunks (splits)
# question = "What are major topics for CS 526?" 
# docs = vectordb.similarity_search(question,k=3)
# print("The length of docs is ", len(docs))

# Step 4: Create LLM
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(model_name=llm_name, temperature=0)
llm.predict("Hello world!")

# Step 5: ConversationalRetrievalChain
# Step 5.1: Create Memory 
from langchain.memory import ConversationBufferMemory
# Step 5.2: QA with ConversationalRetrievalChain
from langchain.chains import ConversationalRetrievalChain
retriever=vectordb.as_retriever()
def load_db():
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        # Set return messages equal true
        # - Return the chat history as a  list of messages 
        #   as opposed to a single string. 
        return_messages=True
    ) 
    qa = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=retriever,
        memory=memory
    )
    return qa


# Step 6: Create a chatbot that works on your documents
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader

# Step 6.1: Create a chatbot that works on your documents ---- Create Business Logic
# def load_db(file, chain_type, k):
    # load documents
    # loader = PyPDFLoader(file)
    # documents = loader.load()
    # # split documents
    # text_splitter = RecursiveCharacterTextSplitter(
    #        chunk_size=1000, 
    #        chunk_overlap=150)
    # docs1 = text_splitter.split_documents(documents)
    # # define embedding
    # embeddings = OpenAIEmbeddings()
    # create vector database from data
    # db = DocArrayInMemorySearch.from_documents(docs1, 
    #        embeddings)
    # define retriever
    # retriever = db.as_retriever(search_type="similarity", 
    #        search_kwargs={"k": k})
    # create a chatbot chain. Memory is managed externally.
    # qa = ConversationalRetrievalChain.from_llm(
    #     llm=ChatOpenAI(model_name=llm_name, temperature=0), 
    #     chain_type=chain_type, 
    #     retriever=retriever, 
    #     return_source_documents=True,
    #     return_generated_question=True,
    # )
    # return qa 


import param
# Step 7.1.2: cbfs class
class cbfs(param.Parameterized):
    chat_history = param.List([])
    answer = param.String("")
    db_query  = param.String("")
    db_response = param.List([])
    
    # Step 7.1.2.1: init function
    def __init__(self,  **params):
        super(cbfs, self).__init__( **params)
        self.panels = []
        self.qa = load_db()

    # Step 7.1.2.2: call_load_db function
    def call_load_db(self, count):
        self.clr_history()
        self.qa = load_db()

    # Step 7.1.2.3: convchain(self, query) function
    def convchain(self, query):
        result = self.qa({"question": query, 
                          "chat_history": self.chat_history})
        self.chat_history.extend([(query, result["answer"])])
        return result["answer"]

    # Step 7.1.2.7: clr_history function
    def clr_history(self):
        self.chat_history = []
        self.qa = load_db()
        return