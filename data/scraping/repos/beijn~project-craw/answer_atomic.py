from lib.Chain import Chain

from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser


instructions = """Given a question and a piece of context, decide whether the context contains the answer to the question. And answer the question using the context."""

class Output(BaseModel):
    found  : bool = Field(description='Whether the answer could be found (or a reasonable estimate made) or not.')
    answer : str  = Field(description='The answer in case the answer was found. If no answer was found the empty string.')




def chain(llm):
    parser = PydanticOutputParser(pydantic_object=Output)

    def res(question, context):
        query = f"\nQuestion:\n{question}\n\Context:\n{context}"

        prompt = f"{instructions}\n{parser.get_format_instructions()}\n{query}\n"

        reply = parser.parse(llm('answer_atomic')(prompt))

        return reply

    return res

