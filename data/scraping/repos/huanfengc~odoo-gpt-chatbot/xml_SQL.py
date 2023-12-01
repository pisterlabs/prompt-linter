import openai
import json
import psycopg2
import tiktoken

db = 'ogpt2'
port = '5432'
username = 'postgres'
password = 'admin' # ALTER USER postgres WITH PASSWORD 'admin';

openai.api_key = ""

conn = psycopg2.connect(database=db,
                        host='localhost',
                        user=username,
                        password=password,
                        port=port)


def get_foreign_key_relations(conn, table_name):
    cursor = conn.cursor()
    query = """
        SELECT
            conname AS constraint_name,
            att2.attname AS column_name,
            cl.relname AS referenced_table,
            att.attname AS referenced_column
        FROM
            pg_constraint AS con
            JOIN pg_class AS cl ON con.confrelid = cl.oid
            JOIN pg_attribute AS att ON con.confrelid = att.attrelid AND con.confkey[1] = att.attnum
            JOIN pg_attribute AS att2 ON con.conrelid = att2.attrelid AND con.conkey[1] = att2.attnum
        WHERE
            con.conrelid = (SELECT oid FROM pg_class WHERE relname = %s)
            AND contype = 'f'
    """
    cursor.execute(query, (table_name,))
    relations = cursor.fetchall()
    cursor.close()
    return relations


def ask_database(conn, query):
    try:
        cursor = conn.cursor()
        cursor.execute(query)
        query_results = cursor.fetchall()
        return query_results
    except Exception as e:
        error_message = str(e)  # Get the error message as a string
        return error_message

def get_table_names(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT table_name FROM information_schema.tables WHERE table_schema='public' AND table_type='BASE TABLE';")
    table_names = cursor.fetchall()
    table_names = [table_name[0] for table_name in table_names]
    return table_names

def get_column_names(conn, table_name):
    cursor = conn.cursor()
    cursor.execute("SELECT column_name FROM information_schema.columns WHERE table_name = %s", (table_name,))
    column_names = cursor.fetchall()
    column_names = [column_name[0] for column_name in column_names]
    return column_names

# SELECT
#     field_description.name AS field_name,
#     field_description.field_description AS field_description,
#     field_description.ttype AS field_type,
#     field_description.relation AS relation,
#     model.model AS model_name
# FROM
#     ir_model AS model
# JOIN
#     ir_model_fields AS field_description
#     ON model.id = field_description.model_id
# WHERE
#     model.model = 'sale.order';
def get_model_cols(conn, model):
    cursor = conn.cursor()
    cursor.execute("""
        SELECT
            field_description.name AS field_name,
            field_description.field_description AS field_description,
            field_description.ttype AS field_type,
            field_description.relation AS relation,
            model.model AS model_name
        FROM
            ir_model AS model
        JOIN
            ir_model_fields AS field_description
            ON model.id = field_description.model_id
        WHERE
            model.model = %s
    """, (model,))
    model_fields = cursor.fetchall()
    cursor.close()

    # turn into csv format
    first_line = "column_name,description,type,related_model\n"

    # field[3] may be none turn into str
    for field in model_fields:
        first_line += field[0] + "," + field[1]["en_US"] + "," + field[2] + "," + str(field[3]) + "\n"

    return first_line


def get_model_fields(conn, list_of_models):
    database_info = []
    for model in list_of_models:
        table_name = model.replace(".", "_")
        # column_names = get_column_names(conn, table_name)
        column_names = get_model_cols(conn, model)
        database_info.append({"table_name": table_name, "column_names": column_names})
    database_info_schema_str = json.dumps(database_info) 
    return database_info_schema_str

models = get_table_names(conn)
models_str = ",".join(models)

encoding = tiktoken.encoding_for_model("gpt-3.5-turbo-0613")
num_tokens = len(encoding.encode(models_str))

def read_id(conn, model, id, fields):
    cursor = conn.cursor()
    table_name = model.replace(".", "_")
    fields_str = ",".join(fields)
    cursor.execute(f"SELECT {fields_str} FROM {table_name} WHERE id = {id}")
    return cursor.fetchall()

functions = [
    {
        "name": "get_model_fields",
        "description": "Use this function to get additional information (columns/fields) about certain models in odoo. Use this function if you aren't sure about a models columns/fields.",
        "parameters": {
            "type": "object",
            "properties": {
                "list_of_models": {
                    "type": "array",
                    "description": f"""
                            Getting additional information about certain models to answer the user's question.
                            """,
                    "items": {
                        "type": "string",
                        "description": "The name of the model to get additional information about.",
                    },
                }
            },
            "required": ["models"],
        },
    },
    {
        "name": "ask_database",
        "description": "You must call get_model_fields before using this. Use this function to answer user questions about Odoo. get_model_fields should always be called before using this. Output should be a fully formed SQL query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": f"""
                            SQL query extracting info to answer the user's question.
                            The query should be returned in plain text, not in JSON.
                            """,
                }
            },
            "required": ["query"],
        },
    },
]

system_prompt = """
    ODOOGPT is a tool that allows you to ask questions about Odoo.
    
    You are Oopo, a friendly AI assistant. 
    
    You are here to help you with your Odoo questions and interact with your Odoo database. 
    
    You should always call get_model_fields before using ask_database. 

    Before I make a CRUD operation on a model I should get additional info about the model and make sure that it indeed exists. 
    You have access to all tables and columns.

    You should never return technical jargon or SQL to the user. 
    You should never say you don't have access to a table or column. Use a function call to find it.
    Try to return names instead of id references

    You have access to all records and information in the database. 
"""

def execute_function_call(message):
    function_name = message["function_call"]["name"]
    print("\033[95m" + "ODOOGPT FUNCTION CALL: " + function_name + "\033[0m")
    args = json.loads(message["function_call"]["arguments"])
    print("\033[95m" + "ODOOGPT FUNCTION ARGUMENTS: " + str(args) + "\033[0m")
    if function_name == "ask_database":
            query = json.loads(message["function_call"]["arguments"])["query"]
            results = ask_database(conn, query)
    elif function_name == "get_model_fields":
        models = json.loads(message["function_call"]["arguments"])["list_of_models"]
        results = get_model_fields(conn, models)
    else:
        results = f"Error: function {message['function_call']['name']} does not exist"

    chat_result = {
        "role": "function",
        "name": message["function_call"]["name"],
        "content": str(results),
    }

    print("\033[95m" + "ODOOGPT FUNCTION RESULT: " + str(results) + "\033[0m")

    return chat_result

def run_conversation():
    total_output_tokens = 0
    total_input_tokens = 0
    
    messages = [
        {"role": "system", "content": system_prompt}
    ]

    while True:
        # check if previous message isn't a function call
        if messages[-1]["role"] != "function":
            prompt = input("Ask Oopo a question (type 'exit' to end the conversation): ")
            
            # Check if the user wants to exit
            if prompt.lower() == "exit":
                break
        
            # Add user message to the messages list
            messages.append({"role": "user", "content": prompt})

        # Call the Chat API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=messages,
            functions=functions,
            function_call="auto",  
        )

        # Get the response message
        if response["choices"][0]["message"].get("function_call"):
            response_message = execute_function_call(response["choices"][0]["message"])
        else:
            response_message = response["choices"][0]["message"]
            print("\033[1m" + response_message["content"] + "\033[1m")

        token_count = response['usage']['total_tokens']
        total_output_tokens = response['usage']['completion_tokens']
        total_input_tokens = response['usage']['prompt_tokens']

        input_cost = total_output_tokens / 1000 * 0.0015
        output_cost = total_input_tokens / 1000 * 0.002

        print("\033[92m" + f"Total cost: ${input_cost + output_cost:.6f} ({token_count})" + "\033[0m")
        print("\033[92m" + f"Total Input cost: ${input_cost:.6f} ({total_input_tokens})" + "\033[0m")
        print("\033[92m" + f"Total Output cost: ${output_cost:.6f} ({total_output_tokens})" + "\033[0m")


        messages.append(response_message)

    return "Conversation ended."

print(run_conversation())
