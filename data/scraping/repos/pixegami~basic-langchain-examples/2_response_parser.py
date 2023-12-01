from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.chat_models.openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser


class Movie(BaseModel):
    title: str = Field(description="The title of the movie.")
    genre: list[str] = Field(description="The genre of the movie.")
    year: int = Field(description="The year the movie was released.")


llm = ChatOpenAI()
parser = PydanticOutputParser(pydantic_object=Movie)

prompt_template_text = """
Response with a movie recommendation based on the query:\n
{format_instructions}\n
{query}
"""

format_instructions = parser.get_format_instructions()
prompt_template = PromptTemplate(
    template=prompt_template_text,
    input_variables=["query"],
    partial_variables={"format_instructions": format_instructions},
)

prompt = prompt_template.format(query="A 90s movie with Nicolas Cage.")
text_output = llm.predict(prompt)
parsed_output = parser.parse(text_output)
print(parsed_output)

# Using LCEL
chain = prompt_template | llm | parser
response = chain.invoke({"query": "A 90s movie with Nicolas Cage."})
print(response)
