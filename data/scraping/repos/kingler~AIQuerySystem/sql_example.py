
import openai
import os
import requests
from tenacity import retry, wait_random_exponential, stop_after_attempt
from termcolor import colored
import requests

import sqlite3

GPT_MODEL = "gpt-3.5-turbo-0613"

# defining the api key from environment variable or inserting it directly
openai.api_key = os.getenv("OPENAI_API_KEY")
# openai.api_key = "sk-<your key here>"

# Defining a function that sends a request to the OpenAI API for chat completion
@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def chat_completion_request(messages, functions=None, model=GPT_MODEL):
    # Setting headers for the request
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai.api_key,
    }
    # Creating a JSON object with the request data
    json_data = {"model": model, "messages": messages}
    # Adding functions to the JSON object if they are provided
    if functions is not None:
        json_data.update({"functions": functions})
    try:
        # Sending a POST request to the OpenAI API with the JSON object as the payload
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        return response
    except Exception as e:
        # If there is an exception, print an error message and return the exception
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


# Defining a Conversation class
class Conversation:
    # Initializing the conversation history as an empty list
    def __init__(self):
        self.conversation_history = []

    # Defining a method to add a message to the conversation history
    def add_message(self, role, content):
        message = {"role": role, "content": content}
        self.conversation_history.append(message)

    # Defining a method to display the conversation history
    def display_conversation(self):
        # Defining a dictionary to map roles to colors for display
        role_to_color = {
            "system": "red",
            "user": "green",
            "assistant": "blue",
            "function": "magenta",
        }
        # Iterating through the conversation history and displaying each message with its role and color
        for message in self.conversation_history:
            print(
                colored(
                    f"{message['role']}: {message['content']}\n\n",
                    role_to_color[message["role"]],
                )
            )

conn = sqlite3.connect("data/Chinook.db")
print("Opened database successfully")


def get_table_names(conn):
    """Return a list of table names."""
    table_names = []
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
    for table in tables.fetchall():
        table_names.append(table[0])
    return table_names


def get_column_names(conn, table_name):
    """Return a list of column names."""
    column_names = []
    columns = conn.execute(f"PRAGMA table_info('{table_name}');").fetchall()
    for col in columns:
        column_names.append(col[1])
    return column_names


def get_database_info(conn):
    """Return a list of dicts containing the table name and columns for each table in the database."""
    table_dicts = []
    for table_name in get_table_names(conn):
        columns_names = get_column_names(conn, table_name)
        table_dicts.append({"table_name": table_name, "column_names": columns_names})
    return table_dicts

# This line calls the function get_database_info() and stores the returned list of dictionaries in the variable database_schema_dict
database_schema_dict = get_database_info(conn)

# This line creates a string representation of the database schema by iterating through the list of dictionaries returned by get_database_info() and formatting the table name and column names for each table
database_schema_string = "\n".join(
    [
        f"Table: {table['table_name']}\nColumns: {', '.join(table['column_names'])}"
        for table in database_schema_dict
    ]
)

# Notice that we are inserting the database schema into the function specification. This will be important for the model to know about.
functions = [
    {
        "name": "ask_database",
        "description": "Use this function to answer user questions about music. Output should be a fully formed SQL query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": f"""
                            SQL query extracting info to answer the user's question.
                            SQL should be written using this database schema:
                            {database_schema_string}
                            The query should be returned in plain text, not in JSON.
                            """,
                }
            },
            "required": ["query"],
        },
    }
]

# This function takes a SQL query and executes it on the provided SQLite database connection object.

def ask_database(conn, query):
    """
    Function to query SQLite database with provided SQL query.
    
    Parameters:
    conn (sqlite3.Connection): A connection object to the SQLite database.
    query (str): A string containing the SQL query to execute on the database.
    
    Returns:
    list: A list of tuples containing the results of the query.
    
    Raises:
    Exception: If there is an error executing the SQL query.
    """
    try:
        # Execute the SQL query on the provided SQLite database connection object and fetch all the results
        results = conn.execute(query).fetchall()
        return results
    except Exception as e:
        # If there is an error executing the SQL query, raise an exception with the error message
        raise Exception(f"SQL error: {e}")


# This function makes a ChatCompletion API call and if a function call is requested, executes the function
# This function takes a list of messages and an optional list of functions as input, makes a ChatCompletion API call using the messages and functions, and if a function call is requested in the response, executes the function and returns the result.
def chat_completion_with_function_execution(messages, functions=None):
    """
    This function makes a ChatCompletion API call and if a function call is requested, executes the function.
    
    Parameters:
    messages (list): A list of strings representing the conversation history.
    functions (list): An optional list of dictionaries representing the functions that can be called by the model.
    
    Returns:
    dict: A dictionary containing the response from the ChatCompletion API call or the result of the function call.
    """
    try:
        # Make a ChatCompletion API call using the messages and functions
        response = chat_completion_request(messages, functions)
        # Get the full message from the response
        full_message = response.json()["choices"][0]
        # If a function call is requested in the response
        if full_message["finish_reason"] == "function_call":
            print(f"Function generation requested, calling function")
            # Call the function and return the result
            return call_function(messages, full_message)
        else:
            print(f"Function not required, responding to user")
            # Return the response from the ChatCompletion API call
            return response.json()
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        # Return the response from the ChatCompletion API call
        return response

# This is the end of the function definition.
# This function executes function calls using model generated function arguments.
def call_function(messages, full_message):
    """Executes function calls using model generated function arguments."""

    # We'll add our one function here - this can be extended with any additional functions
    if full_message["message"]["function_call"]["name"] == "ask_database":
        # Extract the query from the function call arguments
        query = eval(full_message["message"]["function_call"]["arguments"])
        print(f"Prepped query is {query}")
        try:
            # Execute the query using the ask_database function
            results = ask_database(conn, query["query"])
        except Exception as e:
            print(e)

            # If there is an error in the query, try to fix it with a subsequent call
            messages.append(
                {
                    "role": "system",
                    "content": f"""Query: {query['query']}
The previous query received the error {e}. 
Please return a fixed SQL query in plain text.
Your response should consist of ONLY the SQL query with the separator sql_start at the beginning and sql_end at the end""",
                }
            )
            # Retry with the fixed SQL query. If it fails a second time, exit.
            response = chat_completion_request(messages, model="gpt-4-0613")
            try:
                # Extract the SQL query from the response
                cleaned_query = response.json()["choices"][0]["message"]["content"].split("sql_start")[1]
                # Remove the "sql_end" separator from the end of the query
                cleaned_query = cleaned_query.split("sql_end")[0]
                # Print the cleaned query for debugging purposes
                print(cleaned_query)
                # Execute the cleaned query using the ask_database function
                results = ask_database(conn, cleaned_query)
                # Print the results for debugging purposes
                print(results)
                print("Got on second try")
            except Exception as e:
                print("Second failure, exiting")

                print(f"Function execution failed")
                print(f"Error message: {e}")

        # Append the results to the messages list
        messages.append(
            {"role": "function", "name": "ask_database", "content": str(results)}
        )

        try:
            # Make a ChatCompletion API call using the updated messages list
            response = chat_completion_request(messages)
            return response.json()
        except Exception as e:
            print(type(e))
            print(e)
            raise Exception("Function chat request failed")
    else:
        raise Exception("Function does not exist and cannot be called")
    
# This is a string that will be used as the first message from the assistant to the user.
# It introduces the assistant and explains what it does.
agent_system_message = """You are ChinookGPT, a helpful assistant who gets answers to user questions from the Chinook Music Database.
Provide as many details as possible to your users
Begin!"""

# This creates a new Conversation object to keep track of the conversation history.
sql_conversation = Conversation()

# This adds the system message to the conversation history with the role "system".
sql_conversation.add_message("system", agent_system_message)

# This adds the user's first message to the conversation history with the role "user".
sql_conversation.add_message(
    "user", "Hi, who are the top 5 artists by number of tracks"
)


# This line calls the function chat_completion_with_function_execution() and passes in the conversation history and the list of functions as arguments. 
# The returned response is stored in the variable chat_response.
chat_response = chat_completion_with_function_execution(
    sql_conversation.conversation_history, functions=functions
)

# This try block attempts to extract the assistant's message from the chat_response variable.
# If successful, the message is stored in the variable assistant_message and printed to the console.
# If unsuccessful, the error message and the chat_response variable are printed to the console.
try:
    assistant_message = chat_response["choices"][0]["message"]["content"]
    print(assistant_message)
except Exception as e:
    print(e)
    print(chat_response)

# This line adds the assistant's message to the conversation history with the role "assistant".
sql_conversation.add_message("assistant", assistant_message)

# This line displays the conversation history with detailed information.
# If the detailed parameter is set to False, only the messages will be displayed without any additional information.
sql_conversation.display_conversation()

sql_conversation.add_message(
    "user", "What is the name of the album with the most tracks"
)

chat_response = chat_completion_with_function_execution(
    sql_conversation.conversation_history, functions=functions
)
assistant_message = chat_response["choices"][0]["message"]["content"]
print(assistant_message)

