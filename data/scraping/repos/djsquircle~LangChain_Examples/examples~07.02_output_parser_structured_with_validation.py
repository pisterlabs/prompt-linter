# Import standard libraries
from typing import List

# Import third-party libraries
from dotenv import load_dotenv
from pydantic import BaseModel, Field, validator
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

# Load environment variables
load_dotenv()

# Define the data structure to store character traits and their reasoning
class CharacterTraits(BaseModel):
    traits: List[str] = Field(description="List of character traits for a fantasy role")
    reasons: List[str] = Field(description="The reasoning behind each character trait")

    # Validator to ensure that traits do not start with numbers
    @validator('traits')
    def not_start_with_number(cls, field):
        for item in field:
            if item[0].isnumeric():
                raise ValueError("The trait cannot start with numbers!")
        return field
    
    # Validator to ensure that reasons end with a period
    @validator('reasons')
    def end_with_dot(cls, field):
        for idx, item in enumerate(field):
            if item[-1] != ".":
                field[idx] += "."
        return field

# Initialize the parser with the defined data structure
output_parser = PydanticOutputParser(pydantic_object=CharacterTraits)

# Define the template for the language model prompt
prompt_template = """
Offer a list of character traits for a '{character_role}' in a fantasy story based on the provided background and reasoning for each trait.
{format_instructions}
character_role={character_role}
background={background}
"""

# Create a PromptTemplate object with the specified template and variables
prompt = PromptTemplate(
    template=prompt_template,
    input_variables=["character_role", "background"],
    partial_variables={"format_instructions": output_parser.get_format_instructions()}
)

# Background information for the character role 'Wise Wizard'
background_info = "The character is a centuries-old wizard who has guided heroes and kings through countless quests and wars. He is known for his immense knowledge of ancient spells and a calm demeanor even in the face of great danger."

# Format the prompt with the character role and background information
model_input = prompt.format_prompt(
    character_role="Wise Wizard",
    background=background_info
)

# Initialize the OpenAI language model
language_model = OpenAI(model_name='text-davinci-003', temperature=0.7)

# Request the model to generate character traits based on the formatted prompt
model_output = language_model(model_input.to_string())

# Parse the model's output
parsed_output = output_parser.parse(model_output)

# Display the suggested traits for the Wise Wizard
print("Suggested traits for a Wise Wizard:")
for trait, reason in zip(parsed_output.traits, parsed_output.reasons):
    print(f"Trait: {trait}, Reason: {reason}")
