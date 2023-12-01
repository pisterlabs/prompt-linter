# @author     : Jackson Zuo
# @time       : 10/5/2023
# @description: Create ConversationalRetrievalChain according to the prompt.

import os
import openai
from dotenv import load_dotenv
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.memory import ConversationSummaryMemory, ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.vectorstores import FAISS
from werkzeug.utils import secure_filename

from config import Config

load_dotenv()

# openai api keys
openai.api_key = os.getenv('OPENAI_API_KEY')

# sde jobs template
# template = """You are an AI assistant specifically tasked with finding matching
# job opportunities in our job data based on user requests. Your main job is helping
# users find a matching job in the training data.
#
# If the user wants to know the specific company information about the positions or
# if the user wants to know how to apply, tell the user to contact our recruiter
# YQZUO via yqzuo97@gmail.com
#
# AI's name is YUQUE.
# ###
# Some chat pattern examples you can follow:
# AI: Hi! What can I help you with?
# USER: I want to find a job.
# AI: Of course! Ask me about what you are looking for like "Do you have SDE jobs near Boston?"
# USER: Do you have Java Engineer openings near NJ?
# AI: Yes. Here are a few companies in or near New Jersey that may be looking for Java Engineers:
#
# Company A: Company A has offices in New York, NY, and Jersey City, NJ.
#
# Company B: Company B has locations in Johnston, Rhode Island, Phoenix, Arizona, and Iselin,
# New Jersey.
#
# Please note that the availability of positions may vary, and it's always a good idea to
# contact our recruiter YQZUO via yqzuo97@gmail.com
# ###
# Context from data: {context}
# ###
# {chat_history}
#
# Human: {question}
# Chatbot:"""

# nurse jobs template used before
template2 = """You are an AI assistant specifically tasked with finding matching
job opportunities in our job data based on user requests. Your main job is helping 
users find a matching job in the training data. Please provide a concise answer.

If the user wants to know the specific company information about the positions or 
if the user wants to know how to apply, tell the user to contact our recruiter 
YQZUO via yqzuo97@gmail.com

###
Some chat pattern examples you can follow:
AI: Hi! What can I help you with?
USER: I want to find a job.
AI: Of course! Ask me about what you are looking for like "Do you have RN/LPN positions near Boston?"
USER: Do you have RN openings near NJ?
AI: Yes. Here are a few companies in or near New Jersey that may be looking for RN:

Company A: Company A has offices in New York, NY, and Jersey City, NJ.

Company B: Company B has locations in Johnston, Rhode Island, Phoenix, Arizona, and Iselin,
New Jersey. 

Please note that the availability of positions may vary, and it's always a good idea to
contact our recruiter YQZUO via yqzuo97@gmail.com
###
Context from data: {context}
###
{chat_history}

Human: {question}
Chatbot:"""

# template in use
template3 = """You are an AI assistant specifically tasked with finding fit
jobs. Use the following context based and user requests, present any jobs 
that might fit user's requirement. Do not answer or make up positions that 
are not in the context. If there is no match just admit it.

###
Context from data: {context}

###Example
USER: Do you have RN openings near NJ?
AI: I have information about RN openings in New Jersey (NJ). Here are some job opportunities for you:
(job information contract from context)

For more details please contact our recruiter via xxx@xenonhealth.com

###
{chat_history}

Human: {question}
Chatbot:"""

prompt = PromptTemplate(
    input_variables=["question", "chat_history", "context"],
    template=template3
)

# condense_template = """use the follow up input as the output.Don't change any character.For example.
# input: I want to find a job as LPN
# Standalone question: I want to find a job as LPN
# Follow Up Input: {question}
# Standalone question:"""
# condense_question_prompt = PromptTemplate.from_template(condense_template)

embeddings = OpenAIEmbeddings()
# use relative path to avoid path error
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "..", "..", Config.FAISS_INDEX_PATH)
db = FAISS.load_local(file_path, embeddings)


class NoOpLLMChain(LLMChain):
    """No-op LLM chain."""

    def __init__(self):
        """Initialize."""
        super().__init__(llm=OpenAI(), prompt=PromptTemplate(template="", input_variables=[]))

    async def arun(self, question: str, *args, **kwargs) -> str:
        return question

    def run(self, question: str, *args, **kwargs) -> str:
        return question


def getqa():
    """

    Returns: ConversationalRetrievalChain

    """
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    memory = ConversationSummaryMemory(
        llm=OpenAI(temperature=0), memory_key="chat_history", return_messages=True)

    llm = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.2, streaming=True)

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=db.as_retriever(),
        combine_docs_chain_kwargs={"prompt": prompt},
        memory=memory,
        # condense_question_prompt=condense_question_prompt,
        verbose=True
    )
    qa.question_generator = NoOpLLMChain()
    return qa


# helper function used in Dialogflow
def get_session_id(data):
    """
    Extract the session ID from a JSON data object.

    Args:
        data: JSON

    Returns: str: The session ID

    """
    session_info = data.get('sessionInfo', {})
    session_string = session_info.get('session', '')

    parts = session_string.split('/')
    if len(parts) >= 2:
        last_part = parts[-1]
        session_id = last_part.replace("dfMessenger-", '')
        return session_id
    else:
        return None
