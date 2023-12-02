from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langchain.llms import OpenAI
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path="../../.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

llm = OpenAI(model_name="text-davinci-003", openai_api_key=OPENAI_API_KEY)

# How you would like your response structured. This is basically a fancy prompt template
response_schemas = [
    ResponseSchema(name="bad_string", description="This a poorly formatted user input string"),
    ResponseSchema(name="good_string", description="This is your response, a reformatted response")
]

# How you would like to parse your output
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
# See the prompt template you created for formatting
format_instructions = output_parser.get_format_instructions()
print (format_instructions)

template = """
What are the top {list_length} most popular {topic} in the {location}?

{format_instructions}

YOUR RESPONSE:
"""

prompt = PromptTemplate(
    input_variables=["topic", "location", "list_length"],
    partial_variables={"format_instructions": format_instructions},
    template=template
)

promptValue = prompt.format({
    "topic": "sports",
    "location": "United States",
    "list_length": 10
})
_input = prompt.format_prompt(topic="sports", location="United States", list_length=10)

print(promptValue)


llm_output = llm(promptValue)
llm_output

print(output_parser.parse(llm_output))