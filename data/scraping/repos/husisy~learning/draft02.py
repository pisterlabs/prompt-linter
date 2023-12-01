import os
import json
import dotenv
import openai
import requests
import sqlite3
import tenacity
import termcolor
# from tenacity import retry, wait_random_exponential, stop_after_attempt

dotenv.load_dotenv()
openai.api_key = os.environ["OPENAI_API_KEY"]

GPT_MODEL = "gpt-3.5-turbo-0613"


@tenacity.retry(wait=tenacity.wait_random_exponential(min=1, max=40), stop=tenacity.stop_after_attempt(3))
def chat_completion_request(messages, functions=None, function_call=None, model=GPT_MODEL):
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + openai.api_key,
    }
    json_data = {"model": model, "messages": messages}
    if functions is not None:
        json_data.update({"functions": functions})
    if function_call is not None:
        json_data.update({"function_call": function_call})
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=json_data,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e


def pretty_print_conversation(messages):
    role_to_color = {
        "system": "red",
        "user": "green",
        "assistant": "blue",
        "function": "magenta",
    }
    formatted_messages = []
    for message in messages:
        if message["role"] == "system":
            formatted_messages.append(f"system: {message['content']}\n")
        elif message["role"] == "user":
            formatted_messages.append(f"user: {message['content']}\n")
        elif message["role"] == "assistant" and message.get("function_call"):
            formatted_messages.append(f"assistant: {message['function_call']}\n")
        elif message["role"] == "assistant" and not message.get("function_call"):
            formatted_messages.append(f"assistant: {message['content']}\n")
        elif message["role"] == "function":
            formatted_messages.append(f"function ({message['name']}): {message['content']}\n")
    for formatted_message in formatted_messages:
        print(termcolor.colored(formatted_message, role_to_color[messages[formatted_messages.index(formatted_message)]["role"]]))


functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                },
            },
            "required": ["location", "format"],
        },
    },
    {
        "name": "get_n_day_weather_forecast",
        "description": "Get an N-day weather forecast",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "format": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The temperature unit to use. Infer this from the users location.",
                },
                "num_days": {
                    "type": "integer",
                    "description": "The number of days to forecast",
                }
            },
            "required": ["location", "format", "num_days"]
        },
    },
]

messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "What's the weather like today"})
chat_response = chat_completion_request(messages, functions=functions)
assistant_message = chat_response.json()["choices"][0]["message"]
messages.append(assistant_message)
assistant_message
# {'role': 'assistant', 'content': 'Sure, could you please provide me with your location?'}


messages.append({"role": "user", "content": "I'm in Glasgow, Scotland."})
chat_response = chat_completion_request(messages, functions=functions)
assistant_message = chat_response.json()["choices"][0]["message"]
messages.append(assistant_message)
assistant_message
# {
#     'role': 'assistant',
#     'content': None,
#     'function_call': {'name': 'get_current_weather',
#     'arguments': '{\n  "location": "Glasgow, Scotland",\n  "format": "celsius"\n}'}
# }


messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "what is the weather going to be like in Glasgow, Scotland over the next x days"})
chat_response = chat_completion_request(messages, functions=functions)
assistant_message = chat_response.json()["choices"][0]["message"]
messages.append(assistant_message)
assistant_message
# {'role': 'assistant', 'content': 'Sure, I can help you with that. Please provide the number of days you would like to forecast.'}


messages.append({"role": "user", "content": "5 days"})
chat_response = chat_completion_request(messages, functions=functions)
chat_response.json()["choices"][0]
# {
#     'index': 0,
#     'message': {'role': 'assistant',
#     'content': None,
#     'function_call': {'name': 'get_n_day_weather_forecast',
#     'arguments': '{\n  "location": "Glasgow, Scotland",\n  "format": "celsius",\n  "num_days": 5\n}'}},
#     'finish_reason': 'function_call'
# }


# in this cell we force the model to use get_n_day_weather_forecast
messages = []
messages.append({"role": "system", "content": "Don't make assumptions about what values to plug into functions. Ask for clarification if a user request is ambiguous."})
messages.append({"role": "user", "content": "Give me a weather report for Toronto, Canada."})
# function_call=None
# function_call='none'
chat_response = chat_completion_request(
    messages, functions=functions, function_call={"name": "get_n_day_weather_forecast"}
)
chat_response.json()["choices"][0]["message"]
# {
#     'role': 'assistant',
#     'content': None,
#     'function_call': {'name': 'get_n_day_weather_forecast',
#     'arguments': '{\n  "location": "Toronto, Canada",\n  "format": "celsius",\n  "num_days": 1\n}'}
# }



# https://www.sqlitetutorial.net/sqlite-sample-database/
# wget https://www.sqlitetutorial.net/wp-content/uploads/2018/03/chinook.zip

conn = sqlite3.connect('data/chinook.db')

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


database_schema_dict = get_database_info(conn)
database_schema_string = "\n".join(
    [
        f"Table: {table['table_name']}\nColumns: {', '.join(table['column_names'])}"
        for table in database_schema_dict
    ]
)

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

def ask_database(conn, query):
    """Function to query SQLite database with a provided SQL query."""
    try:
        results = str(conn.execute(query).fetchall())
    except Exception as e:
        results = f"query failed with error: {e}"
    return results

def execute_function_call(message):
    if message["function_call"]["name"] == "ask_database":
        query = json.loads(message["function_call"]["arguments"])["query"]
        results = ask_database(conn, query)
    else:
        results = f"Error: function {message['function_call']['name']} does not exist"
    return results


messages = []
messages.append({"role": "system", "content": "Answer user questions by generating SQL queries against the Chinook Music Database."})
messages.append({"role": "user", "content": "Hi, who are the top 5 artists by number of tracks?"})
chat_response = chat_completion_request(messages, functions)
assistant_message = chat_response.json()["choices"][0]["message"]
messages.append(assistant_message)
if assistant_message.get("function_call"):
    results = execute_function_call(assistant_message)
    messages.append({"role": "function", "name": assistant_message["function_call"]["name"], "content": results})
pretty_print_conversation(messages)

messages.append({"role": "user", "content": "What is the name of the album with the most tracks?"})
chat_response = chat_completion_request(messages, functions)
assistant_message = chat_response.json()["choices"][0]["message"]
messages.append(assistant_message)
if assistant_message.get("function_call"):
    results = execute_function_call(assistant_message)
    messages.append({"role": "function", "content": results, "name": assistant_message["function_call"]["name"]})
pretty_print_conversation(messages)
