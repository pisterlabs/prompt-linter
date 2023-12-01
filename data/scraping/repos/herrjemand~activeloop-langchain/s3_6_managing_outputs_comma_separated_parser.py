from dotenv import load_dotenv
load_dotenv(dotenv_path='.env')
import os

from langchain.output_parsers import CommaSeparatedListOutputParser

parser = CommaSeparatedListOutputParser()

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

template = """
Offer a list of suggestions to substitute the word '{target_word}' based the presented the following text: {context}.
{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=['target_word', 'context'],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

model_input = prompt.format(
    target_word="behaviour",
    context="The behaviour of the students in the classroom was disruptive and made it difficult for the teacher to conduct the lesson."
)

model = OpenAI(model_name="text-davinci-003", temperature=0)
output = model(model_input)
result = parser.parse(output)

print(result)