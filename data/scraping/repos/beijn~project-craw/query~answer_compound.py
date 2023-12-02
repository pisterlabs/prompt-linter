from lib.Chain import Chain

from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser


### Your task is first to sanitize and input query into a most likely intended properly specific and well formed question (called 'parent question').
instructions = """Given a question, a reasoning about how the question can be decomposed into simpler subquestions and answers to each of the subquestions deduce the answer.
"""

class Output(BaseModel):
    found  : bool  = Field(description='Whether the answer to the question could be found.')
    answer : str   = Field(description='The answer to the question.')
    explain: str = Field(description='A refined reasoning of how to answer the question using the given answers to the subquestion. Be as brief as possible and avoid obvious arguments and do not reapeat questions or answers. Make it like hints for a smart person. Your are not required to form valid sentences in order be as short as possible.')


def chain(llm):
    parser = PydanticOutputParser(pydantic_object=Output)

    def res(question, reasoning, subanswers):
        query = f"\nQuestion:\n{question}\n\nReasoning:\n{reasoning}"
        for a in subanswers: query += f"\n\nSubanswer:\n{a}"

        prompt = f"{instructions}\n{parser.get_format_instructions()}\n{query}\n"

        return parser.parse(llm('answer_compound')(prompt))

    return res