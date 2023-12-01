from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser, CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.vectorstores import Chroma
from langchain.vectorstores.base import VectorStoreRetriever
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain, TransformChain, SequentialChain

from pydantic import BaseModel, Field, validator
from typing import List
from dotenv import dotenv_values
from utils import process_pdf

env = dotenv_values(".env")
OPEN_AI_API = env['OPEN_AI_API']


def database(splitted_text):
    embeddings = OpenAIEmbeddings(openai_api_key=OPEN_AI_API)
    db = Chroma.from_texts(splitted_text, embeddings)
    return db

pdf = open("notes.pdf", "rb")

splitted_text = process_pdf([pdf])
db = database(splitted_text)


output_parser = CommaSeparatedListOutputParser()
format_instructions = output_parser.get_format_instructions()


question_prompt = PromptTemplate(
    template="""
    The following text consists of questions, but they are not neatly formatted:
    {questions}
    Your task to parse through the text and extract the questions.
    Provide a comma seperated list of questions
    {output_format_instructions}""",
    input_variables=["questions"],
    partial_variables={"output_format_instructions": format_instructions},  
)

llm = OpenAI(openai_api_key=OPEN_AI_API, temperature=0.0)

question_chain = LLMChain(
    llm=llm,
    # your desired llm 
    prompt=question_prompt,
)

def transform_func(response):
    print(response)
    questions = response['text']
    question_list = output_parser.parse(questions)
    return {"question_list": question_list}

transform_chain = TransformChain(input_variables=["text"], output_variables=["question_list"], transform=transform_func)

# Usage
original_question = """
1. Define classification. Describe the general procedure of classification with a neat diagram.
2. Explain the following algorithms in detail with an example:
I. Bayesian Classifier
II. Nearest Neighbour Classifier
"""

# response = question_chain(original_question)
# print("response: ", response)
# response = output_parser.parse(response['text'])
# print("response: ", response)

overall_chain = SequentialChain(
    chains=[question_chain, transform_chain], 
    input_variables=["questions"],
    verbose=True
)

questions_list = overall_chain({"questions": original_question})['question_list']

for q in questions_list:
    print(q)


# print(type(response['text']))

