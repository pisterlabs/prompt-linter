from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts.chat import (ChatPromptTemplate,
                                    HumanMessagePromptTemplate,
                                    SystemMessagePromptTemplate)

response_schemas = [
    ResponseSchema(name="summary", description="summary of the weather"),
    ResponseSchema(
        name="status",
        description="predicted status of the weather - can be one of: Poor, Fair, Average, Good or Very Good",
    ),
    ResponseSchema(
        name="inspiring-message",
        description="uplifting message about the weather",
    ),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = output_parser.get_format_instructions()

template = """
On the following lines is a CSV representation of the weather forecast. 
The first row contains the column names. 
Use only this data for the summary.
{csv}
-----
Use the following code mappings to map any codes in the data to meaningful labels.
{code_mappings}
-----
{format_instructions}
-----
Summarise the weather for the next few hours as follows. Do not including the datetime:
1. For the summary: Imagine you are a weatherman and summarise the data in no more than 200 words.
2. For the predicted status: It must consider the temperature, chance of rain and weather type.
3. For the inspiring message: It must be inspiring and uplifting. It must be no more than 300 words. It must be appropriate to the predicted status.
"""

human_template = """
Create the summary
"""

chat_prompt = ChatPromptTemplate(
    messages=[
        SystemMessagePromptTemplate.from_template(template),
        HumanMessagePromptTemplate.from_template(human_template),
    ],
    partial_variables={"format_instructions": format_instructions},
    input_variables=["code_mappings", "csv"],
)


def get_prompt():
    return output_parser, chat_prompt
