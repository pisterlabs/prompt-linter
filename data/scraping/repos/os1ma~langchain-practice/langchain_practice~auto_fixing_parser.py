from typing import List

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import OutputFixingParser, PydanticOutputParser
from pydantic import BaseModel, Field
from util import initialize

initialize()


class Actor(BaseModel):
    name: str = Field(description="name of an actor")
    film_names: List[str] = Field(description="list of names of films they starred in")


misformatted = "{'name': 'Tom Hanks', 'film_names': ['Forrest Gump']}"

parser = PydanticOutputParser(pydantic_object=Actor)
# parser.parse(misformatted)


llm = ChatOpenAI(model_name="gpt-3.5-turbo")
new_parser = OutputFixingParser.from_llm(parser=parser, llm=llm)
result = new_parser.parse(misformatted)
print(result)
