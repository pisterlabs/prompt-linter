import openai
import os
import requests
from tenacity import retry,wait_random_exponential,stop_after_attempt
from termcolor import colored
from dotenv import dotenv_values

import sqlite3

GPT_MODEL = "gpt-3.5-turbo-0613"
config = dotenv_values(".env")
openai.api_key= config['OPENAI_API_KEY']



@retry(wait=wait_random_exponential(min=1,max=40),stop=stop_after_attempt(3))
def chat_completion_request(messages,functions=None,model = GPT_MODEL):
    
    headers = {
        "Content-type":"application/json",
        "Authorization":"Bearer" + openai.api_key
    }

    json_data = {"model":model,"messages":messages}

    if functions is not None:
        json_data.update({"functions":functions})

        try:
            # response = openai.ChatCompletion.create(
            #     model=GPT_MODEL,
            #     messages = messages,
            #     functions = functions,
            #     function_call = "auto",
            # )
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
    

class Conversation: 
    def __init__(self) :
        self.conversation_history = []

    def add_message(self,role,content):
        message = {"role":role,"content":content}
        self.conversation_history.append(message)


    def display_conversation(self):

        role_to_color = {
            "system":"red",
            "user":"green",
            "assistant":"blue",
            "function":"magenta"
        }
        for message in self.conversation_history :
            print(
              colored(
                f"{message['role']}:{message['content']}\n\n",
                role_to_color[message["role"]],
              )
            )
    

conn = sqlite3.connect("data\chinook.db")
print ("Database Sucesfully Opened")


def get_table_names(conn):
    """Return a list of table names"""
    table_names = []
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type ='table';")
    for table in tables.fetchall():
        table_names.append(table[0])
    return table_names

def get_column_names(conn,name):
    column_names = []
    columns = conn.execute(f"PRAGMA table_info('{name}');").fetchall()
    for col in columns:
        column_names.append(col[1])
    return column_names

def get_database_info(conn):
    table_dicts = []

    for table_name in get_table_names(conn):
        column_names = get_column_names(conn,table_name)
        table_dicts.append({"table_name":table_name,"column_names":column_names})
    return table_dicts


database_schema_dict = get_database_info(conn)

database_schema_string = "\n".join(
    [
        f"Table: {table['table_name']}\nColumns : {','.join(table['column_names'])}"
        for table in database_schema_dict
    ]
)

functions = [
    {
        "name":"ask_database",
        "description":"Use this function to answer the user questions about music. Output should be a fully formed SQL query.",
        "parameters":{
            "type":"object",
            "properties":{
                "query":{
                    "type":"string",
                    "description":f"""
                            SQL query extracting info to answer the user's question.
                            SQL should be written using this database schema:
                            {database_schema_string}
                            The query should be returned in plain text, not in JSON
                    """
                }
            }
        },
        "required":["query"],
    }
]

def ask_database(conn,query):
    """
    Function to query SQLite database with provided SQL query.

    Parameters:
    conn(sqlite3.Connection)
    query(str)
    """

    try:
        results = conn.execute(query).fetchall()
        
        return results
        
    except Exception as e:
        raise Exception(f"SQL error : {e}")
    

def chat_completion_with_function_execution(messages,functions=None):
    try:
        response = chat_completion_request(messages,functions)

        data = response.json()
       
        full_message = data["choices"][0]

        if full_message["finish_reason"] == "function_call":
            print(f"function generation requested, calling function")
            return call_function(messages,full_message)
        
        else :
            print(f"Function not required, responding to user")
            return response.json()
    except Exception as e:
        print("Unable to generate ChatCompletion Response")
        print(f"Exception : {e}")

        return response
    
def call_function(messages,full_messages):
    if full_messages["message"]["function_call"]["name"] == "ask_database":
        query = eval(full_messages["messaage"]["function_call"]["arguments"])
        print(f"prepped query is {query}")

        try:
            results = ask_database(conn,query["query"])
        
        except Exception as e:
            print(e)

            messages.append(
                {
                    "role":"system",
                    "content":f"""Query: {query['query']}
                    the previous query received the error {e}.
                    Please return a fixed SQL query in plain text.
                    Your response should consist of only the sql query with the separator sql_start at the beginning and sql_end at the end
                    """,
                }
            )
            reponse = chat_completion_request(messages, model="gpt-3.5-turbo")
            try :
                cleaned_query =reponse.json()["choices"][0]["message"]["content"].split("sql_start")[1]

                cleaned_query = cleaned_query.split("sql_end")[0]

                print(cleaned_query)

                results = ask_database(conn,cleaned_query)
                print(results)
                print("Got on second try")
            except Exception as e:
                print("Second Failure, exiting")

                print("Function execution failed")
                print (f"Error Message: {e}")

        messages.append(
            {"role":"function","name":"ask_database","content":str(results)}
        )

        try:
            response = chat_completion_request(messages)
            return response.json()
        except Exception as e:
            print(type(e))
            print(e)
            raise Exception("Function chat request failed")
    else:
        raise Exception("Function does not exist and cannot be called") 


agent_system_message = """You are AG-BOT, a helpful assitant who gets answers to user questions from the Chinook Music Database.
Provide as many details as prossible to your users
"""


sql_conversation = Conversation()

sql_conversation.add_message("system",agent_system_message)

sql_conversation.add_message(
    "user","Hi, who are the top 5 artists by number of tracks"
)

chat_response = chat_completion_with_function_execution(
    sql_conversation.conversation_history,functions=functions
)

try:
    assistant_message = chat_response["choices"][0]["message"]["content"]
    print(assistant_message)
except Exception as e:
    print(e)
    print(chat_response)


sql_conversation.add_message("assistant",assistant_message)

sql_conversation.display_conversation

sql_conversation.add_message(
    "user","What is the name of the album with the most tracks"
)

chat_response = chat_completion_with_function_execution(
    sql_conversation.conversation_history,functions=functions
)

assistant_message = chat_response["choices"][0]["message"]["content"]
print(assistant_message)


