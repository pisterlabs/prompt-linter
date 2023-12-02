from dotenv import load_dotenv
load_dotenv()

from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

# Initialize the output parser
parser = CommaSeparatedListOutputParser()

# Prepare the Prompt
template = """
Offer a list of character traits for a '{character_role}' in a fantasy story based on the provided background: {background}.
{format_instructions}
"""

prompt = PromptTemplate(
    template=template,
    input_variables=["character_role", "background"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

# Example background for a character role 'Wise Wizard'
background = "The character is a centuries-old wizard who has guided heroes and kings through countless quests and wars. He is known for his immense knowledge of ancient spells and a calm demeanor even in the face of great danger."

model_input = prompt.format(
  character_role="Wise Wizard",
  background=background
)

# Loading OpenAI API
model = OpenAI(model_name='text-davinci-003', temperature=0.7)

# Send the Request
output = model(model_input)

# Parse and Print the Response
print("Suggested traits for a Wise Wizard:", parser.parse(output))
