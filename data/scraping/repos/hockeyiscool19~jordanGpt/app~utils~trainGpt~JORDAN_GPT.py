# https://python.langchain.com/docs/use_cases/question_answering/how_to/multi_retrieval_qa_router
import os
from langchain.prompts import PromptTemplate
from langchain.chains.router import MultiRetrievalQAChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from PUBLIC_VARIABLES import OPENAI_API_KEY, FINE_TUNING_JOB
from langchain.chains import LLMChain
import openai
from langchain.chat_models import ChatOpenAI
from app.utils.firebase import FIRE
from datetime import datetime


openai.api_key = OPENAI_API_KEY


FILES = [
    "app/utils/data/resume.txt",
    "app/utils/data/roleDescriptions.txt",
]


resume = TextLoader("app/utils/data/resume.txt").load_and_split()
resume_retriever = FAISS.from_documents(
    resume, OpenAIEmbeddings()).as_retriever()

role = TextLoader("app/utils/data/roleDescriptions.txt").load_and_split()
role_retriever = FAISS.from_documents(role, OpenAIEmbeddings()).as_retriever()


RETRIEVER_INFO = [
    {
        "name": "resume",
        "description": "Answers questions about Jordan Eisenmann's resume",
        "retriever": resume_retriever
    },
    {
        "name": "role descriptions",
        "description": "Describes questions about Jordan Eisenmann's role descriptions",
        "retriever": role_retriever
    },
]

RESPOND_ROLE = """
    You are Jordan Eisenman (Jordan) secretary, answering questions about his career and passions. Be relevant to this prompt. You answer
    questions and use background information to assist. Redirect questions about "you" to Jordan. 
    Question: {question}
    Background information: {retrieved}
    """
RESPOND_ROLE.format(question="what is your name",
                    retrieved="Jordan Eisenmann is my name")

RESPOND_PROMPT = PromptTemplate(template=RESPOND_ROLE, input_variables=[
                                "question", "retrieved"])


class JordanGpt:
    def __init__(self, verbose=True):
        # Initializing Retrieval chain
        self.retriever_chain = MultiRetrievalQAChain.from_retrievers(ChatOpenAI(
            model_name=FINE_TUNING_JOB, max_tokens=125), RETRIEVER_INFO, verbose=verbose)
        # Initializing Response chain
        self.chat = ChatOpenAI(model_name=FINE_TUNING_JOB, max_tokens=175)
        self.respond_role = RESPOND_ROLE
        self.conversation_chain = LLMChain(
            llm=self.chat, verbose=verbose, prompt=RESPOND_PROMPT)

    def logQuestionAnswer(self, question, answer, retrieved):
        data = {
            "messages": [{
                "question": question,
                "answer": answer,
                "prompt": RESPOND_ROLE.format(question=question, retrieved=retrieved),
                "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }]
        }
        FIRE.load_dict(data, path='/jordanGpt/trainingData')

    def logRetrieved(self, retrieved):
        data = {"log": retrieved,
                "time": datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        FIRE.load_dict(data, path='/logs')

    def retrieve(self, question):
        retrieved = self.retriever_chain.run(question)
        return retrieved

    def respond(self, question):
        retrieved = self.retrieve(question)
        self.logRetrieved(retrieved)
        response = self.conversation_chain.run(
            {"question": question, "retrieved": retrieved})
        self.logQuestionAnswer(question, response, retrieved)
        return response


JORDAN_GPT = JordanGpt(verbose=False)
