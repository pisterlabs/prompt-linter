import os
from dotenv import load_dotenv
# Use the environment variables to retrieve API keys
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

print("#######################Comma seperated List##############################")
#Comma seperated List
from langchain.output_parsers import CommaSeparatedListOutputParser
output_parser = CommaSeparatedListOutputParser()

format_instructions = output_parser.get_format_instructions()
print(format_instructions)

prompt = PromptTemplate(
    template = "Provide 5 example of {query}.\n{format_instructions}",
    input_variables = ["query"],
    partial_variables = {"format_instructions": format_instructions}
)

llm = OpenAI(temperature = .9, model = "text-davinci-003")
prompt = prompt.format(query="Currencies")
output = llm(prompt)
print(output)

print("#######################Json Format##############################")
#Json Format
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
response_schemas = [
    ResponseSchema(name="currency", description = "Answer to the user question"),
    ResponseSchema(name="abbreviation", description = "what is the abbreviation of that currency")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()
print(format_instructions)

prompt = PromptTemplate(
    template = "Answet to the question best as possible. \n{format_instructions}\n {query}",
    input_variables = ["query"],
    partial_variables = {"format_instructions": format_instructions}
)

prompt = prompt.format(query="What is the currency of india ?")
output = llm(prompt)
print(output)