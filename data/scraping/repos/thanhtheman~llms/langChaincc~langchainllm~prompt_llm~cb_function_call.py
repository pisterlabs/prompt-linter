from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from dotenv import load_dotenv
load_dotenv()

"""
It is a nightmare to work with unstructured output from LLMs. Reges is not fun.
We need to parse the output (which will be used downstream) into a structured format.
"""

prompt = ChatPromptTemplate.from_template("tell a joke about a {role}")
model = ChatOpenAI()

functions = [
    {
        "name": "joke",
        "description": "A joke",
        "parameters": {
        "type": "object",
        "properties": {
            "setup": {
            "type": "string",
            "description": "The setup for the joke"
            },
            "punchline": {
            "type": "string",
            "description": "The punchline for the joke"
            }, 
            "explanation": {
                "type":"string",
                "description": "The explanation for the joke" #this is the extra prompt we will use to get an explanation for the joke
            }                                                 #it is not included in the original prompt
        },
        "required": ["setup", "punchline", "explanation"]
        }
    }
]

chain = prompt | model.bind(function_call={"name": "joke"}, functions=functions) | JsonOutputFunctionsParser()
response = chain.invoke({"role": "doctor"})
print(response)