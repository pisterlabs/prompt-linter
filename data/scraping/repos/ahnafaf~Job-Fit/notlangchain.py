import peewee
import utils
import openai
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts.chat import HumanMessagePromptTemplate
from langchain.schema.messages import SystemMessage, HumanMessage
from langchain.utilities import SQLDatabase
from langchain.llms import OpenAI
from langchain_experimental.sql import SQLDatabaseChain
from utils import connect_to_db
import sys
import tiktoken

llm = OpenAI()
chat_model = ChatOpenAI()
model_used = "gpt-3.5-turbo-16k-0613"
encoding = tiktoken.get_encoding("cl100k_base")
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")


def process_raw_job_data_smart(limit):
    llm = ChatOpenAI(temperature=0, verbose=True, max_tokens = 1000, model = model_used)
    db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)
    content = f"""Query the whole table in the latest {limit} number of results. You are an helpful assistant analyzes the jobs and not the table and returns the following information:
    General Analysis, changes from historical trends, top keywords from descriptions (by number of occurrences)"""
    return db_chain.run(content)

def process_raw_job_data(content):
    llm = ChatOpenAI(temperature=1, verbose=True, max_tokens = 1000, model = model_used)
    template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                """You are an helpful assistant that analyzes the jobs given, to which you return the following as statistics in much detail, make educated guesses. Aim for as many words as possible.:
                General Analysis:
                Top Keywords:
                Skills to Watch Out For:
                Top Locations (if applicable):"""
            )
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
    )
    response = llm(template.format_messages(text=content))
    return response

def process_raw_job_data_resume(content, resume):
    llm = ChatOpenAI(temperature=1, verbose=True, max_tokens = 1000, model = model_used)
    template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                """You are an helpful assistant that analyzes the jobs given, to which you return the following as statistics in much detail, make educated guesses. Aim for as many words as possible.:
                General Analysis:
                Top Keywords:
                Skills to Watch Out For:
                Top Locations (if applicable):
                Detailed Analysis of Resume and Comparison to Jobs:"""
            )
        ),
        HumanMessagePromptTemplate.from_template("Jobs: {text}"),
        HumanMessagePromptTemplate.from_template("Resume: {resume}"),
    ]
    )
    response = llm(template.format_messages(text=content, resume=resume))
    return response

def process_continued_job_data(analysis, jobs):
    llm = ChatOpenAI(temperature=0, verbose=True, max_tokens = 1000, model =model_used)
    template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                """Add to the analysis of the jobs, you will be given the analysis and the jobs.
                General Analysis:
                Changes from Historical Trends:
                Top Keywords:
                Top Locations (if applicable):"""
            )
        ),
        HumanMessagePromptTemplate.from_template("Analysis = {analysis}"),
        HumanMessagePromptTemplate.from_template("Jobs = {jobs}")
    ]
    )
    response = llm(template.format_messages(analysis=analysis, jobs=jobs))
    return response


def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

    

"""
def processer():
    # 1st, we need to go ahead and make the processed table.
    readFromDb
    chatHistory = ...
    for i in rowofdb:
        data = []
        counter = 0
        if counter < 3500:
            data.append(i)
            counter += amount_of_chars
        else:
            counter = 0
            processed_data = process_existing_data(data)
            chatHistory += processed_data
            data = []
"""