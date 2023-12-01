from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8)

response_schema=[
    ResponseSchema(name="bad_string", description="this is a poorly string input by the user"),
    ResponseSchema(name="good_string", description="this is your response, a reformatted response.")
]

output_parser = StructuredOutputParser.from_response_schemas(response_schema)

#there is some pre-generated prompt for the instruction to ensure it is a json format, before getting into your schema
format_instructions = output_parser.get_format_instructions()

template = """
You will be given a poorly formatted string from a user.
Reformat it and make sure all the words are spelled correctly

{format_instructions}

% USER INPUT:
{user_input}

YOUR RESPONSE:
"""
#somehow it introduces a partial_variable={"format_instructions": format_instructions}, I don't get it.
# I just make it a normal variable and it works fine.
prompt = PromptTemplate(input_variables=["user_input", "format_instructions"],
                        template=template)
format_prompt = prompt.format(user_input="I a m grom Tornton canada", format_instructions=format_instructions)

# the output of the model is always a AIMessage object, 
full_response = model([HumanMessage(content=format_prompt)])
print(full_response)

#extract the content as string so that it can be parsed
parsed_response = output_parser.parse(full_response.content)
print(parsed_response)