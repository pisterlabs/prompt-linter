import openai
from uuid import uuid4

from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate
import psycopg2
from psycopg2 import sql
import json
from langchain.vectorstores import Chroma
from langchain.document_loaders import JSONLoader
from langchain.document_loaders.csv_loader import CSVLoader

from llm import llm, chat_llm, embeddings
from langchain.memory import ChatMessageHistory
import pandas as pd
from llm import llm, chat_llm
from where_clause import *
from table_description import get_top_10_unique_values

history = ChatMessageHistory()


"""
{'unique_id': '3d2d655e_0f26_42ef_bbf0_8859fe5f6196'}

"""


def get_basic_table_details(cursor):
    cursor.execute("""SELECT
            c.table_name,
            c.column_name,
            c.data_type
        FROM
            information_schema.columns c
        WHERE
            c.table_name IN (
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = 'public'
    );""")
    tables_and_columns = cursor.fetchall()
    return tables_and_columns


def get_foreign_key_info(cursor):
    query_for_foreign_keys = """SELECT
    conrelid::regclass AS table_name,
    conname AS foreign_key,
    pg_get_constraintdef(oid) AS constraint_definition,
    confrelid::regclass AS referred_table,
    array_agg(a2.attname) AS referred_columns
    FROM
        pg_constraint
    JOIN
        pg_attribute AS a1 ON conrelid = a1.attrelid AND a1.attnum = ANY(conkey)
    JOIN
        pg_attribute AS a2 ON confrelid = a2.attrelid AND a2.attnum = ANY(confkey)
    WHERE
        contype = 'f'
        AND connamespace = 'public'::regnamespace
    GROUP BY
        conrelid, conname, oid, confrelid
    ORDER BY
        conrelid::regclass::text, contype DESC;
    """

    cursor.execute(query_for_foreign_keys)
    foreign_keys = cursor.fetchall()

    return foreign_keys



def create_vectors(filename, persist_directory):
    loader = CSVLoader(file_path=filename, encoding="utf8")
    data = loader.load()
    vectordb = Chroma.from_documents(data, embedding=embeddings, persist_directory=persist_directory)
    vectordb.persist()





def save_db_details(db_uri='postgresql://apoorvagarwal@localhost:5432/parceldb'):

    unique_id = str(uuid4()).replace("-", "_")
    connection = psycopg2.connect(db_uri)
    cursor = connection.cursor()

    tables_and_columns = get_basic_table_details(cursor)
    values = []

    ## Get all the tables and columns and enter them in a pandas dataframe
    df = pd.DataFrame(tables_and_columns, columns=['table_name', 'column_name', 'data_type'])
    filename_t = 'csvs/tables_' + unique_id + '.csv'
    df.to_csv(filename_t, index=False)

    


    create_vectors(filename_t, "./vectors/tables_"+ unique_id)


    ## Get all the foreign keys and enter them in a pandas dataframe
    foreign_keys = get_foreign_key_info(cursor)
    df = pd.DataFrame(foreign_keys, columns=['table_name', 'foreign_key', 'foreign_key_details', 'referred_table', 'referred_columns'])
    filename_fk = 'csvs/foreign_keys_' + unique_id + '.csv'
    df.to_csv(filename_fk, index=False)

    cursor.close()
    connection.close()

    return unique_id






def gather_information(query, unique_id):

    vectordb = Chroma(embedding_function=embeddings, persist_directory="./vectors/tables_"+ unique_id)
    retriever = vectordb.as_retriever()
    docs = retriever.get_relevant_documents(query)
    print(docs)

    relevant_tables = []
    relevant_tables_and_columns = []


    for doc in docs:
        table_name, column_name, data_type = doc.page_content.split("\n")
        table_name= table_name.split(":")[1].strip()
        relevant_tables.append(table_name)
        column_name = column_name.split(":")[1].strip()
        data_type = data_type.split(":")[1].strip()
        relevant_tables_and_columns.append((table_name, column_name, data_type))


    ## Load the tables csv
    filename_t = 'csvs/tables_' + unique_id + '.csv'
    df = pd.read_csv(filename_t)

    ## For each relevant table create a string that list down all the columns and their data types
    table_info = ''
    for table in relevant_tables:
        table_info += 'Information about table' + table + ':\n'
        table_info += df[df['table_name'] == table].to_string(index=False) + '\n\n\n'

    
    ## Load the foreign keys csv
    filename_fk = 'csvs/foreign_keys_' + unique_id + '.csv'
    df_fk = pd.read_csv(filename_fk)



    ## If table from relevant_tables above lies in refered_table or table_name in df_fk, then add the foreign key details to a string

    foreign_key_info = ''
    extra_tables = []
    for i, series in df_fk.iterrows():
        if series['table_name'] in relevant_tables:
            text = table + ' has a foreign key ' + series['foreign_key'] + ' which refers to table ' + series['referred_table'] + ' and column(s) ' + series['referred_columns']
            foreign_key_info += text + '\n\n'
            extra_tables.append(series['referred_table'])
        if series['referred_table'] in relevant_tables:
            text = table + ' is referred to by table ' + series['table_name'] + ' via foreign key ' + series['foreign_key'] + ' and column(s) ' + series['referred_columns']
            foreign_key_info += text + '\n\n'
            extra_tables.append(series['referred_table'])
    
    other_tables = list(set(extra_tables) - set(relevant_tables))
    additional_table_info = ''
    for table in other_tables:
        additional_table_info += 'Information about table ' + table + ':\n'
        additional_table_info += df[df['table_name'] == table].to_string(index=False) + '\n\n\n'

        

    return relevant_tables, relevant_tables_and_columns, table_info, foreign_key_info, additional_table_info



def generate_template_for_sql(query, relevant_tables, table_info, foreign_key_info, additional_table_info):
    tables = ",".join(relevant_tables)
    template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        f"You are an assistant that can write SQL Queries."
                        f"Given the text below, write a SQL query that answers the user's question."
                        f"Assume that there is/are SQL table(s) named '{tables}' "
                        f"Here is a more detailed description of the table(s): "
                        f"{table_info}"
                        "Here is some information about some relevant foreign keys:"
                        f"{foreign_key_info}"
                        "If in doubt which tables and columns to use, ask the user for more information."
                        "Prepend and append the SQL query with three backticks '```'"
                        
                        
                    )
                ),
                HumanMessagePromptTemplate.from_template("{text}"),

            ]
        )
    
    answer = chat_llm(template.format_messages(text=query))
    print(answer.content)
    return answer.content
    


def check_if_users_query_want_general_schema_information_or_sql(query):
    template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        
                        f"In the text given text user is asking a question about database "
                        f"Figure out whether user wants information about database schema or wants to write a SQL query"
                        f"Answer 'yes' if user wants information about database schema and 'no' if user wants to write a SQL query"
                        
                    )
                ),
                HumanMessagePromptTemplate.from_template("{text}"),

            ]
        )
        
    answer = chat_llm(template.format_messages(text=query))
    print(answer.content)
    return answer.content


    
    

def prompt_when_user_want_general_db_information(query, db_uri):
    template = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=(
                        "You are an assistant who writes SQL queries."
                        "Given the text below, write a SQL query that answers the user's question."
                        "Prepend and append the SQL query with three backticks '```'"
                        "Write select query whenever possible"
                        f"Connection string to this database is {db_uri}"
                    )
                ),
                HumanMessagePromptTemplate.from_template("{text}"),

            ]
        )
    
    answer = chat_llm(template.format_messages(text=query))
    print(answer.content)
    return answer.content


    
    




def execute_the_solution(db_uri, solution):

    connection = psycopg2.connect(db_uri)
    cursor = connection.cursor()
    _,final_query,_ = solution.split("```") 
    final_query = final_query.strip('sql')
    cursor.execute(final_query)
    result = cursor.fetchall()
    return result







def complete_process(query, unique_id, db_uri):

    answer_to_question_general_schema = check_if_users_query_want_general_schema_information_or_sql(query)

    if answer_to_question_general_schema == "yes":
        solution = prompt_when_user_want_general_db_information(query, db_uri)
        result = execute_the_solution(db_uri, solution)
        return result

    relevant_tables, relevant_tables_and_columns, table_info, foreign_key_info, additional_table_info = gather_information(query, unique_id)

    solution = generate_template_for_sql(query, relevant_tables, table_info, foreign_key_info, additional_table_info)

    result = execute_the_solution(db_uri, solution)

    print("*"*10)
    print(len(result))
    print(result)
    print("*"*10)

    ### check if result contains any rows

    if len(result) ==  0 or result is None or result[0] is None or result[0][0] is None:
        # return solution, result

        if if_where_in_solution(solution):
            all_column_value_info = gather_all_column_information(query, solution, unique_id, db_uri, relevant_tables_and_columns)
            print(all_column_value_info)
            solution = generate_template_for_sql_with_where_clause(query, relevant_tables, table_info, foreign_key_info, additional_table_info, all_column_value_info)
            result = execute_the_solution(db_uri, solution)
            return result


    result = execute_the_solution(db_uri, solution)
    return result
    






        

        











"""

from querying import *
from where_clause import *
unique_id = 'a3a30875_3df4_453e_832d_49fb43409cda'
db_uri = 'postgresql://apoorvagarwal@localhost:5432/parceldb'
query = "what are agricultural dlc areas and their rates in rajasthan?"
relevant_tables, relevant_tables_and_columns, table_info, foreign_key_info, additional_table_info = gather_information(query, unique_id)
solution = generate_template_for_sql(query, relevant_tables, table_info, foreign_key_info, additional_table_info)
all_column_value_info = gather_all_column_information(query, solution, unique_id, db_uri, relevant_tables_and_columns)
execute_the_solution(db_uri, solution)

"""