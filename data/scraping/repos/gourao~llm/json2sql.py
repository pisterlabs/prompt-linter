import os
import re
import sys
import environ
import pdb
import json
import jsonschema

from urllib.parse import urlparse

import sqlite3
import psycopg2

from jsonschema import Draft7Validator

from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

if len(sys.argv) < 3:
	print("Missing  args.  Example 'json2sql schema.json <path to data> postgresql+psycopg2://postgres:1234@localhost:6667/mydb'")
	exit()

def trim_(input_string):
	last_underscore_index = input_string.rfind("_\d+")
	if last_underscore_index != -1:  # If underscore is found
		return input_string[:last_underscore_index]
	else:
		return ""

# Breath first so the parent tables get created before children
def create_schema(cursor, schema, table_name=''):
	parent_table_name = trim_(table_name)
	print("Processing JSON for : ", table_name, parent_table_name)

	column_types = {}
	child_schemas = {}
	child_id = 0

	if isinstance(schema, dict):
		# If JSON data is a dictionary, recursively process its values
		for key, value in schema.items():
			key = re.sub(r'[^a-zA-Z0-9]', '_', key).lower()

			if isinstance(value, (dict, list)):
				# For nested dictionaries create a separate table with the key in the table name
				child_schemas[table_name + "_" + str(child_id) + "_" + key] = value
				child_id = child_id + 1
			else:
				# Insert primitive values as a column
				if value == "int":
					column_types[key] = 'INTEGER'
				elif value == "float":
					column_types[key] = 'REAL'
				elif value == "bool":
					column_types[key] = 'BOOL'
				else:
					column_types[key] = 'TEXT'
	elif isinstance(schema, list):
		# If JSON data is a list, iterate over its elements
		for item in schema:
			if isinstance(item, (dict, list)):
				# For nested lists, create a separate table with a 0 in the table name to indicate a list
				child_schemas[table_name + "_" + str(child_id)] = item
				child_id = child_id + 1
			else:
				# Insert primitive values as a column
				item = re.sub(r'[^a-zA-Z0-9]', '_', item).lower()

				if value == "int":
					column_types[item] = 'INTEGER'
				elif value == "float":
					column_types[item] = 'REAL'
				elif value == "bool":
					column_types[item] = 'BOOL'
				else:
					column_types[item] = 'TEXT'

	# Create SQL table if there are any primitive columns to add
	drop_table_sql = f"DROP TABLE IF EXISTS {table_name} CASCADE;"
	cursor.execute(drop_table_sql)

	# Needed for GPT-3.5... override any existing ID field in the 
	# original data so we can create a relationship
	if column_types.get("id") is not None:
		print("Overriding existing ID of type ", column_types["id"])
	column_types["id"] = 'INTEGER PRIMARY KEY'

	create_table_sql = f"CREATE TABLE {table_name} ("
	create_table_sql += ", ".join([f"{key} {column_types[key]}" for key in column_types])
	if parent_table_name != "" :
		create_table_sql += f", FOREIGN KEY (id) REFERENCES {parent_table_name}(id)"
	create_table_sql += ");"

	print("Executing: ", create_table_sql)
	cursor.execute(create_table_sql)

	# Create child schemas
	for key, value in child_schemas.items():
		create_schema(cursor, value, key)

def import_data(cursor, json_data_path):
	pass

json_data_path = sys.argv[1]
if json_data_path == "":
	print("Missing document path")
	exit()

schema_file_path = sys.argv[2]
if schema_file_path == "":
	print("Missing schemat path")
	exit()

# Read JSON Schema
with open(schema_file_path, 'r') as schema_file:
	schema = json.load(schema_file)

# Print the JSON schema
print(json.dumps(schema, indent=2))

# Create SQL schema and import data

db_string = sys.argv[3]
if db_string == "":
	print("Missing DB string")
	exit()

db_url = urlparse(db_string)

# Create a connection to the PostgreSQL server
conn = psycopg2.connect(
	dbname=db_url.path[1:],
	user=db_url.username,
	password=db_url.password,
	host=db_url.hostname,
	port=db_url.port
)

print("Connected to PostgreSQL!")

# Create a cursor to execute SQL queries
cursor = conn.cursor()

# Execute a sample query (replace with your SQL queries)
cursor.execute("SELECT version();")
db_version = cursor.fetchone()
print(f"PostgreSQL Database Version: {db_version[0]}")

create_schema(cursor, schema, "main")
conn.commit()
prompt = input("Schema created")

import_data(cursor, json_data_path)
prompt = input("Data imported into SQL")
