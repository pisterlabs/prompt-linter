from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain.schema.runnable import RunnableMap, RunnablePassthrough

from dotenv import load_dotenv
load_dotenv()

"""
we can even make the code shorter, by letting the variable runs through the prompt.
variables value -> prompt -> model -> parser -> text answer
"""
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
# map_ = RunnableMap(role=RunnablePassthrough())
prompt = ChatPromptTemplate.from_template("tell a joke about a {role}")
model = ChatOpenAI()

chain = ({"role": RunnablePassthrough()} | prompt | model.bind(function_call={"name": "joke"}, functions=functions) | JsonOutputFunctionsParser())
response = chain.invoke({"stupid librarian"})
print(response)