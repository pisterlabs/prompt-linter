import os
import sys
import shutil

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

class DocumentChat:

  def __init__(self):
    # Enable to save to disk & reuse the model (for repeated queries on the same data)
    self.PERSONAL_DOCS_FOLDER = "documents"
    self.PERSIST_FOLDER = self.PERSONAL_DOCS_FOLDER + "_persist"
    # GPT_MODEL = "gpt-3.5-turbo-16k"
    self.GPT_MODEL = "gpt-3.5-turbo"

    self.chat_history = []
    self.build_embedding_chain(self.PERSIST_FOLDER, self.PERSONAL_DOCS_FOLDER, clean=False)
  
  # ====================================================================================================

  def ask_conversational_question(self, question, clear_chat_history=False): 
      # print("\n=========================================\nAsking conversational question:\n" + 
      #       question + "\n=========================================\n")
      if (clear_chat_history):
        self.chat_history = []
      
      answer = self.chain({"question": question, "chat_history": self.chat_history})['answer']
      self.chat_history.append((question, answer))
      return answer

  # ====================================================================================================

  def get_answer(self, question):
    return self.index.query(question)

  # ====================================================================================================

  def build_embedding_chain(self, persist_folder, personal_docs_folder, clean=False):

    if (clean):
      print("Rebuilding the personal docs index...")
      # if the persist_folder exists, delete it
      if os.path.exists(persist_folder):
        shutil.rmtree(persist_folder)
      # create a new index
      loader = DirectoryLoader(personal_docs_folder)
      self.index = VectorstoreIndexCreator(
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100), vectorstore_kwargs={"persist_directory":self.PERSIST_FOLDER}).from_loaders([loader])
    else:
      print("Reusing vectorstore from " + self.PERSIST_FOLDER + " directory...\n")
      vectorstore = Chroma(persist_directory=self.PERSIST_FOLDER, embedding_function=OpenAIEmbeddings())
      self.index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    
    self.chain = ConversationalRetrievalChain.from_llm(
      llm=ChatOpenAI(model=self.GPT_MODEL),
      retriever=self.index.vectorstore.as_retriever(search_kwargs={"k": 10}),
    )

  # ====================================================================================================

  def start(self):

    while True:
      print("\n=====================================================\n" +
            "Ask a question about the documents Enter 'Q = Quit', 'R' = Refresh Documents" +
            "\n=====================================================\n")
      userInput = input("Q: ")
      if userInput in ['Q', 'Quit']:
        break
      elif userInput in ['R', 'Refresh'] :
        index = self.build_embedding_chain(self.PERSIST_FOLDER, self.PERSONAL_DOCS_FOLDER, clean=True)
        chat_history = []
      else:
        print("\n... Generating answer ...\n")
        answer = self.ask_conversational_question(userInput)
        print(answer + "\n")

  # END OF CLASS ResumeBuilder ====================================================================================================

DocumentChat().start()
