import json
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv,find_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from termcolor import colored
from llm_logger import logger
#Setup the logging configuration
load_dotenv(find_dotenv())


llm = ChatOpenAI(temperature=0, model="gpt-4")

template = """
You are a super intelligent Python grader which will rate Python code based on the quality of the code. 

Here is the rubric for the criteria and the weights of each Python grade:
Correctness: Does the code function as intended? (60)
Efficiency: Is the code optimized for performance? (20)
Style: Does the code adhere to the coding conventions and standards? (10)
Modularity and Organization: Is the code well-organized and modular? (10)

Given the problem and solution below, assign a score from 0-100

Problem:
{problem}

Solution: 
{solution}
"""

prompt = PromptTemplate(
    input_variables=["problem", "solution"],
    template=template
)


grade_template = '''
Print just the score out of 100 as one number.

{graded_response}
'''

grade = PromptTemplate(
    input_variables=["graded_response"],
    template=grade_template
)

chain = LLMChain(llm=llm, prompt=prompt)
grade_chain = LLMChain(llm=llm, prompt=grade)

def truncate_text(text, max_length):
    if len(text) > max_length:
        return text[:max_length - 3] + '...'
    return text

def grade_code(problem, solution):
    reasoning = chain.run(problem=problem, solution=solution)

    logger.info(f"The solution has been grading. Extracting the final number")
    final_grade = grade_chain.run(graded_response=reasoning)

    formatted_problem = truncate_text(problem, 80)
    print(formatted_problem)

    print(colored(f'Final Grade: {final_grade}/100', 'yellow'))

    return final_grade



