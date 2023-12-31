from aiModules import openAIBase as oai
import json
from aiModules.templates import ticketOutputTemplates as tot
from langchain.chat_models import ChatOpenAI

oai.setAIKeyFromEnvVariables()
local_ai = oai.ai

def getFunctionDefinition(functionString, format_style):
    name = f"get{functionString}Template"
    description = f"In {format_style} style, get the output for ticket contents based on the provided context"
    parameters = {
        "type": "object",
        "properties": {
            "conversation": {
                "type": "object",
                "description": "The conversation that has taken place between the user and the AI so far",
            },
            "ticket_type": {
                "type": "string",
                "description": "The type of ticket being written"
            },
            "result_type": {
                "type": "string",
                "description": "The type of result being generated by the ticket"
            },
            "format_type": {
                "type": "string",
                "description": "The format of the result being generated by the ticket"
            }
        },
        "required": ["conversation"],
    }
    
    return {
        "name": name,
        "description": description,
        "parameters": parameters,
    }

def getTemplateFromFunctionCall(conversation):
    messages = []
    for c in conversation:
        if c.type == 'human':
            role = 'user'
        elif c.type == 'ai':
            role = 'assistant'
        else:
            role = 'system'
        messages.append({"role": role, "content": f"{c.content}"})
    functions = [
        getFunctionDefinition("Gherkin", "gherkin"),
        getFunctionDefinition("Markdown", "markdown"),
        getFunctionDefinition("PlainText", "plain text"),
        getFunctionDefinition("SqlScript", "sql script")
    ]
    response = local_ai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto",
    )
    response_message = response["choices"][0]["message"]

    if response_message.get("function_call"):
        available_functions = {
            "getGherkinTemplate": tot.getGherkinTemplate,
            "getMarkdownTemplate": tot.getMarkdownTemplate,
            "getPlainTextTemplate": tot.getPlainTextTemplate,
            "getSqlScriptTemplate": tot.getSqlScriptTemplate
        }
        function_name = response_message["function_call"]["name"]
        function_to_call = available_functions[function_name]
        function_args = json.loads(response_message["function_call"]["arguments"])
        print(function_args)
        function_response = function_to_call(conversation=conversation, **function_args)

        messages.append(response_message)
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )
        print(messages)
        return runTemplateFromFunctionCall(function_response)
    
def runTemplateFromFunctionCall(func_response):
    chat = ChatOpenAI(openai_api_key=oai.ai.api_key)
    print(func_response)
    response = chat(func_response)
    print(response)
    return response