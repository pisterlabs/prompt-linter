from langchain.prompts import ChatPromptTemplate
from langchain.llms import Ollama
import json

def perform_operation(params:dict):
    """Extract information from the dict to evaluate the simple operation"""
    op = str(params["first_parameter"]) + params["operator"] + str(params["second_parameter"])
    print("Operation:", op, "; Result:", eval(op))


# Prompt template
template_string = """You are a world class algorithm for extracting information from text in structured formats.
Extract the information from this operation: {message}{format}
"""

FORMAT_INSTRUCTIONS = """
The output should be formatted as a single JSON instance that conforms to the JSON examples below. Do not add any sentence, only the json.

{
    "operator": "+",
    "first_parameter": value, 
    "second_parameter": value
}

{
    "operator": "/",
    "first_parameter": value, 
    "second_parameter": value
}

{
    "operator": "*",
    "first_parameter": value, 
    "second_parameter": value
}

"""
# Initialize model
llm = Ollama(model="zephyr", 
             temperature=0)
print("The operation shoud be like 'What is 10 plus 10?'") # Instructions for the User

while True:
    # Initialize the prompt template
    prompt = ChatPromptTemplate.from_template(template_string)
    messages = prompt.format_messages(message= input(">>"), format=FORMAT_INSTRUCTIONS)

    try:
        # And execute the model
        output = llm(messages[0].content)
        output = output.replace("json", "").replace("```", "") # Clean up the model generation (when using zephyr it displays those labels attached to the json)
        print(output) 
        parameters = json.loads(output) # Transforms the string generated from the model in an actual json
        perform_operation(parameters) # And perfom the operation

    except:
        print("Operation failed, try making the input more similar to the example")


