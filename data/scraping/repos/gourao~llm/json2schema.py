import os
import sys
import environ
import pdb
import json
import jsonschema
from jsonschema import Draft7Validator

from pysondb import db

from langchain import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

env = environ.Env()
environ.Env.read_env()

# Load API key and document location
OPENAI_API_KEY = env('OPENAI_API_KEY')

if OPENAI_API_KEY == "":
	print("Missing OpenAPI key")
	exit()

print("Using OpenAPI with key ["+OPENAI_API_KEY+"]")

path = sys.argv[1]
if path == "":
	print("Missing document path")
	exit()

# For Schema
with open(path, 'r') as json_file:
	json_data = json.load(json_file)

# Extract the schema
def extract_schema(data):
    if isinstance(data, dict):
        schema = {}
        for key, value in data.items():
            schema[key] = extract_schema(value)
        return schema
    elif isinstance(data, list):
        if len(data) > 0:
            # Assuming all elements in the list have the same schema
            return [extract_schema(data[0])]
        else:
            return []
    else:
        return type(data).__name__

schema = extract_schema(json_data)

# Print the JSON schema
print(json.dumps(schema, indent=2))

prompt = input("Hit Enter to Continue")

# For in mem JSON DB
d = db.getDb(path)

print(d.getAll())

# prompt loop
def get_prompt():
	print("Type 'exit' to quit")

	while True:
		prompt = input("Enter a prompt: ")

		if prompt.lower() == 'exit':
			print('Exiting...')
			break
		else:
			try:
				pass
			except Exception as e:
				print(e)

get_prompt()
