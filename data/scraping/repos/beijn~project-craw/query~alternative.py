from lib.Chain import Chain

from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser


### Your task is first to sanitize and input query into a most likely intended properly specific and well formed question (called 'parent question').
instructions = """Given a list of subquestions and reasons why they could not be answered by an external system, propose a new question that is more likely to be answerable by the external system. The external system is usually a simple fact checker on wikipedia that can only lookup facts explicitely written in wikipedia. That subquestion could not be answered means, you should propose an alternative more general question that can be used to estimate the original question. For example the question "What is the albedo of Barack Obama?" couldn't be answered directly using wikipedia, your job is to propose the question "What is the ethnicity of Barack Obama?" with the explanation that the albedo can be estimated from the ethnicity (low for afroamerican people, high for people of norteuropean descendance). The new question has to be closely related to the subquestions! If you cannot come up with a related estimatory alternative question simply repeat the first given question.
"""

class Output(BaseModel):
    question : str = Field(description='A question that is similar to the given subquestions.')
    explain  : str = Field(description='How the new question relates to the common denominator of the given subquestion.')


def chain(llm):
    parser = PydanticOutputParser(pydantic_object=Output)

    def res(questions):
        query = f""
        for uaq in questions: query += f"\nSubquestion:\n{uaq.question}\n\nReason why it could not be answered:\n{uaq.answer}"

        prompt = f"{instructions}\n{parser.get_format_instructions()}\n{query}\n"

        return parser.parse(llm('alternative')(prompt))

    return res