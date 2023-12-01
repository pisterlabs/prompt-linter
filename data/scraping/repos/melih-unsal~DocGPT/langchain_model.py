from dotenv import load_dotenv
import os
load_dotenv()
from langchain.document_loaders import TextLoader
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI

class LangchainModel:
    def __init__(self,filepath='/home/melih/Downloads/Introduction to User Research_Course_Book.txt'):
        if filepath.endswith('.txt'):
            loader = TextLoader(filepath)
        else:
            loader = PyPDFLoader(filepath)
        self.index = VectorstoreIndexCreator().from_loaders([loader])
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k",temperature=0)
    
    def answer(self,query):
        answer = self.index.query(query,llm=self.llm)
        print(answer)
        if "i don't know" in answer.lower():
            answer = self.llm(query)
            answer = "Document-based answer has failed. Question is asked to standard model...\n"+"."*100+"\n" + answer

        return answer

