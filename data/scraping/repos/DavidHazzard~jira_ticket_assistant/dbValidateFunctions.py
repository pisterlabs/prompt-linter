import os
import re
import sqlalchemy as sa
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate,PromptTemplate,SystemMessagePromptTemplate,HumanMessagePromptTemplate,AIMessagePromptTemplate
from databaseModules.dbConnects import databaseValidate as dv

## Constants ##
### AI object/global variable init ###
ai_key = os.getenv("OPENAI_API_KEY_GPT4")
db_lc = ChatOpenAI(openai_api_key=ai_key, temperature="0.0", verbose=False)
tables_in_use = []
table_schemas = []

### Table validate prompt templates ###
query_table_list_template = "Output a comma-delimited list of the tables referenced in the following query: {query}. Do not output any natural language."
predict_table_name_template = "Use the following schema to predict the correct table name for {table_name}: {database_table_names}. Output the correct table name for {table_name}. Only output the table name."

### Column validate prompts templates ###
query_column_dict_template = "Output the columns explicitly referred to in the following query: {query}. Remove table aliases from each column, and do not include any duplicate fully qualified column names. If a column name is not specified, do not include it in the output. Do not output any natural language. Use the following template to format the data: {column_dict_template}"
column_dict_output_format = "{ table: [ columns ] }"
predict_column_name_template = "Use the following schema to predict the correct column name for {column}: {table_columns}. Output the correct column name for {column}. Only output the column name. Do not output any natural language."

## Prompt Templating Functions ##
def getHumanMessagePrompt(template):
    return HumanMessagePromptTemplate.from_template(template)

def getPrompt(human_template):
    return ChatPromptTemplate.from_messages([getHumanMessagePrompt(human_template)])

def formatPrompt(chat_prompt, input_query, input_table_name="", input_database_table_names="", input_column_dict_template="", input_column="", input_table_columns=""):
    return chat_prompt.format_prompt(query=input_query,
                                     table_name=input_table_name,
                                     database_table_names=input_database_table_names,
                                     column_dict_template=input_column_dict_template,
                                     column=input_column,
                                     table_columns=input_table_columns).to_messages()

## Globally used functions ##
### Table Query Parsing Functions ###
def get_table_list_from_query(query):
    table_list = db_lc(
        formatPrompt(
            getPrompt(query_table_list_template)
            ,query
        )
    ).content.split(",")
    return [x.strip().lower() for x in table_list]

def get_schemas_for_tables_in_use():
    global table_schemas

    if table_schemas != []:
        return table_schemas
    
    table_list_processed =  {table.name.lower(): table for table in dv.db.metadata.tables.values()}
    print(table_list_processed)
    if len(tables_in_use) > 0:
        table_list_processed = {key: value for key, value in table_list_processed.items() if key in tables_in_use}
    for table in table_list_processed:
        print(f"\n {table}")
        table_schemas.append(get_table_metadata(table_list_processed[table]))
    return table_schemas

### Table Metadata Retrieval Functions ###
def get_table_metadata(table):
    return {
        "name": table.name,
        "primary_key": get_primary_key(table),
        "foreign_keys": get_fk_relationships(table),
        "columns": get_columns(table)
    }
    
def get_columns(table):
    return [c.name.lower() for c in table.columns]

def get_fk_relationships(table):
    fk_columns = [fk.parent for fk in table.foreign_keys]
    for column in fk_columns:
        return column.name.lower(), column.foreign_keys
    
def get_primary_key(table):
    return [k.name for k in table.primary_key]

## Table Validation Functions ##
def validateTables(query):
    global tables_in_use
    global table_schemas

    # get list of all table names in database
    database_table_names = [x.lower() for x in dv.db.inspector.get_table_names()]
    
    # get list of all table names being used in query
    query_table_list = get_table_list_from_query(query)

    # validate each table in query
    for table_name in query_table_list:
        # if table is not in database, handle invalid query
        if database_table_names.count(table_name) == 0:
            query = handleInvalidQuery(query, table_name, database_table_names)
        # otherwise, add table to global list of tables in use
        else:
            print(f"{table_name} is valid.\n")
            tables_in_use.append(table_name) if tables_in_use.count(table_name) == 0 else None
    print(tables_in_use)

    # get table schemas for all tables in use (global variable)
    table_schemas = get_schemas_for_tables_in_use()

    return sa.text(query)

def handleInvalidQuery(query, table_name, database_table_names):
    # Print error message
    print(f"Invalid query. The table '{table_name}' does not exist in the database.")

    # Prompt the user to enter a new table name
    predict_table_name = db_lc(
        formatPrompt(
            getPrompt(predict_table_name_template)
            ,input_query=""
            ,input_table_name=table_name
            ,input_database_table_names=database_table_names
        )
    ).content

    # Replace the original table name with the user's new table name
    query = re.sub(table_name, predict_table_name, query, 1, re.IGNORECASE)

    # Print message to indicate the table name replacement
    print(f"Replaced {table_name} with {predict_table_name} \n")

    # Add the new table name to the global list of tables in use
    tables_in_use.append(predict_table_name) if tables_in_use.count(predict_table_name) == 0 else None

    return query


## Column Validation Functions ##
def validateColumns(query):
    global table_schemas
    global tables_in_use

    # If no tables have been specified, get them from the query.
    if len(tables_in_use) == 0:
        tables_in_use = get_table_list_from_query(query)

    # Get a dictionary of tables and columns from the query.
    query_column_dict = get_query_column_dict(query)

    print(f"Query columns by table: \n {query_column_dict} \n")

    # Get the table schemas for the tables in use.
    table_schemas = get_schemas_for_tables_in_use()

    # For each table in the table schema, check if each column in the query
    # exists in the table schema. If a column does not exist in a table,
    # replace it with an empty string.
    for table in table_schemas:
        for column in query_column_dict[table['name'].lower()]:
            if table['columns'].count(column.lower()) == 0:
                print(f"Invalid query. The column '{column}' does not exist in the table '{table['name']}'.")
                query = replace_invalid_column(query, column, table['columns'])
            else:
                print(f"{column} is valid.\n")
    print(query)
    return sa.text(query)

def get_query_column_dict(query):
    # query the db for column names
    query_column_dict = eval(
        db_lc(
            formatPrompt(
                getPrompt(query_column_dict_template)
                ,query
                ,input_column_dict_template=column_dict_output_format
            )  
        ).content
    )
    
    # convert all of the column names to lowercase
    for key in query_column_dict:
        query_column_dict[key] = [x.lower() for x in query_column_dict[key]]

    return query_column_dict

def replace_invalid_column(query, column, table_columns):
    # Get a single column name from the query
    predict_column_name = db_lc(
        formatPrompt(
            getPrompt(predict_column_name_template)
            ,input_query=""
            ,input_column=column
            ,input_table_columns=table_columns
        )
    ).content

    # Replace the invalid column with the AI-predicted column name
    query = re.sub(column, predict_column_name, query, 1, re.IGNORECASE)
    print(f"Replaced '{column}' with '{predict_column_name}' \n")
    return query


## Join Validation Functions ##
def validateJoins(query):
    pass

## Data Type Validation Functions ##
def validateDataTypes(query):
    pass