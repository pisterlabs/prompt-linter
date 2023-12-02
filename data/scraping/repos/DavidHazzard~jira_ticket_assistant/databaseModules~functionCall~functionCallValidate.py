import openai as ai
import os
import json
from databaseModules.dbValidateFunctions import dbValidateFunctions as dvf
from langchain.chat_models import ChatOpenAI
ai.api_key = os.getenv("OPENAI_API_KEY")

def getFunctionDefinition(query_part, qp_natural):
    name = f"validate{query_part}"
    description = f"Using the current database schema, validate the {qp_natural} used in the SQL query."
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The SQL query to be validated"
            }
        },
        "required": ["query"],
    }
    
    return {
        "name": name,
        "description": description,
        "parameters": parameters,
    }


def validateQueryFromFunctionCall(sql_exception, query):
    messages = [{"role": "user", "content": sql_exception}]
    functions = [
        getFunctionDefinition("Tables", "table(s)"),
        getFunctionDefinition("Columns", "column(s)"),
        getFunctionDefinition("DataTypes", "data type(s)")
    ]
    response = ai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        functions=functions,
        function_call="auto",
    )
    response_message = response["choices"][0]["message"]
    print(response_message)

    if response_message.get("function_call") and response_message["function_call"]["name"] != "python":
        available_functions = {
            "validateTables": dvf.validateTables,
            "validateColumns": dvf.validateColumns
            #"validateDataTypes": dvf.validateDataTypes 
            # ## This function is not yet implemented
        }
        function_name = response_message["function_call"]["name"]
        print(function_name)

        function_to_call = available_functions[function_name]
        print(function_to_call)

        function_args = json.loads(response_message["function_call"]["arguments"])
        function_response = function_to_call(**function_args)

        messages.append(response_message)
        messages.append(
            {
                "role": "function",
                "name": function_name,
                "content": function_response,
            }
        )
        print("\nFunction call first response:")
        print(messages)

        print(f"Function results: \n\n {function_response}")
        return function_response
    else:
        return response_message
        
        
