"""
This module provides functionality to get arguments for flight search from user query and 
to find flights using a language model and a SQL database toolkit.
"""

import json
import openai
from langchain.tools import tool
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.llms.openai import OpenAI
from langchain.agents.agent_types import AgentType

def get_args(query_user, OPENAI_KEY):
    """
    Extracts necessary parameters for flight search from user query using OpenAI API.
    
    This function takes a user query and an OpenAI key as inputs, sends the user query to 
    OpenAI API to extract necessary parameters including the number of adults, departure date,
    return date, destination location code, and origin location code for flight search.

    Parameters:
    query_user (str): User query to be sent to OpenAI API.
    openai_key (str): OpenAI key to authenticate with the API.

    Returns:
    num_adults (int): Number of adults for the flight.
    departureDate (str): Departure date in the format YYYY-MM-DD.
    returnDate (str): Return date in the format YYYY-MM-DD.
    destinationLocationCode (str): IATA code for the destination location.
    originLocationCode (str): IATA code for the origin location.
    """

    function_call = [
    {
      "name": "search_for_flights",
      "description": "Requests flight data from Amadeus API and writes to SQLite database",
      "parameters": {
        "type": "object",
        "properties": {
            "num_adults":{
                "type":"integer",
                "description": '''Based on the query, respond with the number of adults'''
            },
            "departureDate": {
                "type":"string",
                "description": '''Based on the query, respond with the Departure Date. Dates are specified in the ISO 8601 YYYY-MM-DD format. '''
            },
            "returnDate": {
                "type":"string",
                "description": '''Based on the query, respond with the Return Date. Dates are specified in the ISO 8601 YYYY-MM-DD format. '''
            },
            "destinationLocationCode":{
                "type":"string",
                "description": '''Based on the query, respond with an airport IATA code from the city which the traveler is going. E.g CDG for Charles de Gaulle Airport'''
            },
          "originLocationCode": {
            "type": "string",
            "description": '''Based on the query, respond with an airport IATA code from the city which the traveler will depart from. E.g CDG for Charles de Gaulle Airport'''
          },

         "TypeofflightReuqest": {
            "type": "string",
            "description": '''Based on the query, respond with the type of flight the user is requesting E.g cheapest, shortest, fastest, least stops etc.'''
          },

        },
        "required": ["destinationLocationCode", "originLocationCode", "departureDate", "returnDate", "num_adults", "TypeofflightReuqest"]
      }
    }
    ]
    
    openai.api_key = OPENAI_KEY

    message = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[{"role": "user", "content": query_user}],
        functions = function_call,
        function_call = 'auto',
        temperature=0
    )
    response_message = message["choices"][0]["message"]["function_call"]["arguments"]

    parsed_data = json.loads(response_message)

    # Accessing variables
    num_adults = parsed_data['num_adults']
    departureDate = parsed_data['departureDate']
    returnDate = parsed_data['returnDate']
    destinationLocationCode = parsed_data['destinationLocationCode']
    originLocationCode = parsed_data['originLocationCode']
    TypeofflightReuqest = parsed_data['TypeofflightReuqest']

    
    print("Number of Adults: ", num_adults)
    print("Departure Date: ", departureDate)
    print("Return Date: ", returnDate)
    print("Destination Location Code: ", destinationLocationCode)
    print("Origin Location Code: ", originLocationCode)
    print("Origin Location Code: ", TypeofflightReuqest)


    return num_adults, departureDate, returnDate, destinationLocationCode, originLocationCode, TypeofflightReuqest


# run SQLDatabase chain
def find_flights(query, llm, db):
    """
    Executes a search for flights using a language model and a SQL database toolkit.
    
    Parameters:
    query (str): The query to be executed, typically a natural language description of the flights to find.
    llm (LanguageModel): The language model used to process the query and generate SQL commands.
    db (Database): The database object where the flight data is stored and from which data will be retrieved.
    
    Returns:
    Response: The response from the agent executor's run method, typically containing the search results or an error message.
    """
        
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)
    agent_executor = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    
    return agent_executor.run(query)