import json
from openai import OpenAI
from describe_database import get_db_schema
from colorama import Fore, Style

with open('db_config.json', 'r') as f:
    db_config = json.load(f)

# Load the API key from api_key.txt
with open('api_key.txt', 'r') as f:
    openai_api_key = f.read().strip()


class InitializationError(Exception):
    pass


# Function to initialize session with GPT-3, make it aware of the database schema, and it's purpose as a Chef Companion
def initialize_session(db_config, openai_api_key):
    client = OpenAI(
        api_key=openai_api_key,
    )
    db_schema = get_db_schema(
        db_config["host"],
        db_config["user"],
        db_config["password"],
        db_config["database"],
    )
    messages = []
    prompt = (
        "You are a SQL Database companion who can execute SQL queries on behalf of the user. Try to ensure"
        "you understand completely what the user is asking for before executing the query."
        f"The database you will be interfacing with is described as such: {db_schema}. Keep your responses short and to"
        f" the point. You aare interfacing with the user through a command line interface. Say hello and describe what"
        f" you understand the structure of the database to be"
    )
    messages.append({"role": "system", "content": f"{prompt}"})
    functions = [
        {
            "type": "function",
            "function": {
                "name": "execute_sql",
                "description": "Execute a SQL query using the app's database context",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql_query": {
                            "type": "string",
                            "description": "The SQL query to execute",
                        },
                    },
                    "required": ["sql_query"],
                },
            },
        }
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0613", messages=messages,
            tools=functions,
            tool_choice="auto"
        )
        replied_message = response.choices[0].message.content
        print(f"{Fore.GREEN}GPT: {replied_message}{Style.RESET_ALL}")
    except Exception as e:
        # Handle API error here, e.g. retry or log
        print(f"OpenAI API returned an Error: {e}")
        raise InitializationError(str(e))

    return [response.choices[0].message], client


db_connection = None


def set_db_connection(connection):
    global db_connection
    db_connection = connection


# Function to execute a SQL query
def execute_sql(sql_query):
    print("Querying Database....")
    global db_connection
    connection = db_connection
    cursor = connection.cursor()
    sql_query = sql_query.lower()
    try:
        cursor.execute(sql_query)
        if sql_query.lower().startswith("select"):
            return str(cursor.fetchall())  # Return fetched data for SELECT queries
        connection.commit()  # Commit changes for INSERT, UPDATE, DELETE queries
        return "Query executed successfully"
    except Exception as e:
        print(f"An error occurred: {e}. Letting GPT know")
        return(f"An error occurred during sql_query: {e}")
    finally:
        cursor.close()


FUNCTION_MAP = {
    "execute_sql": execute_sql,
}


# Function to send input to GPT-3 and parse output
def send_message(messages: list, client):
    functions = [
        {
            "type": "function",
            "function": {
                "name": "execute_sql",
                "description": "Execute a SQL query using the app's database context",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sql_query": {
                            "type": "string",
                            "description": "The SQL query to execute",
                        },
                    },
                    "required": ["sql_query"],
                },
            },
        }
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0613", messages=messages,
            tools=functions,
            tool_choice="auto"
        )
    except Exception as e:
        print(f"OpenAI API returned an Error: {e}"
              f"Message log: {messages}")
        raise InitializationError(str(e))
    response_message = response.choices[0].message
    while response_message.tool_calls:
        messages.append(response_message)
        tool_call = response_message.tool_calls[0]
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle error
        function_name = tool_call.function.name
        tool_call_id = tool_call.id
        function_to_call = FUNCTION_MAP[function_name]
        function_args = [val for key, val in json.loads(tool_call.function.arguments).items()]
        function_response = function_to_call(*function_args)
        # Step 4: send the info on the function call and function response to GPT4
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": function_response,
            }
        )  # extend conversation with function response
        second_response = client.chat.completions.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
            tools=functions,
            tool_choice="auto",
        )  # get a new response from GPT where it can see the function response
        response_message = second_response.choices[0].message

    return response_message.content
