from typing import List

from langchain.llms import OpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field, validator
from typing import List, Optional

model_name = "text-davinci-003"
temperature = 0.0
model = OpenAI(model_name=model_name, temperature=temperature)

# Define your desired data structure.
class bill_item(BaseModel):
    item: str = Field(description="name of item that was sold")
    quantity: int = Field(description="Quantity of item that was sold")
    unit: str = Field(description="unit in which quantity was measured like l kg etc")

    # You can add custom validation logic easily with Pydantic.
    # @validator("setup")
    # def question_ends_with_question_mark(cls, field):
    #     if field[-1] != "?":
    #         raise ValueError("Badly formed question!")
    #     return field

class bill(BaseModel):
    billitem : List[bill_item] = []
    

# And a query intented to prompt a language model to populate the data structure.
def t2j(text: str):
    bill_query = text

    # Set up a parser + inject instructions into the prompt template.
    parser = PydanticOutputParser(pydantic_object=bill)

    prompt = PromptTemplate(
        template="extract the details about the items, quantity and units from the given input with each set of item its quantity and its unit is one set use only short forms like kg, l, instead of the word packet in unit use u.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    _input = prompt.format_prompt(query=bill_query)

    output = model(_input.to_string())
    print(output)
    output = parser.parse(output)
    return output
    


