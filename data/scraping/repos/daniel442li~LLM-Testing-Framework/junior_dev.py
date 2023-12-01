import json
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv,find_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from grader_dev import grade_code, truncate_text
from llm_logger import logger
import re
from termcolor import colored

#Setup the logging configuration
load_dotenv(find_dotenv())



llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")

template = """
You are a super intelligent Python developer. 
You will create a solution based on a JSON. 
The task will be described in a description and there will be two fields: an input and an output that can be used to test the solution. 
Only return the code.

Below is the task:
{task}
"""

prompt = PromptTemplate(
    input_variables=["task"],
    template=template
)
chain = LLMChain(llm=llm, prompt=prompt)

def run_test(file):
    # Load the JSON file
    total_score = 0
    questions = 0
    with open('tests/' + file, 'r') as file:
        data = json.load(file)

    for item in data:
        questions += 1
        logger.info(f"Junior dev is solving the problem: {truncate_text(item['description'], 100)}")
        solution = chain.run(task=item)
        problem = item['description']

        logger.info(f"The problem has been solved! Sending it to the grader.")
        score = grade_code(problem, solution)

        total_score += int(score)

    overall_score = total_score / questions
    print(colored(f'Overall Score: {overall_score}', 'green'))
