import os
import openai
import json
import subprocess
import tempfile
from dataset.simulation import Agent, Simulation
from dataset.dataset import Dataset
from knowledge_representation import get_default_ltmc
from knowledge_representation.knowledge_loader import load_knowledge_from_yaml, populate_with_knowledge

# set API key
openai_keys_file = os.path.join(os.getcwd(), "keys/openai_keys.txt")
with open(openai_keys_file, "r") as f:
    keys = f.read()
keys = keys.strip().split('\n')
os.environ["OPENAI_API_KEY"] = keys[0]

class SQLAgent(Agent):
	initial_prompt = \
"You have access to a PostgreSQL database that describes information about the state of a household. "  \
"You are an agent that will perform tasks around the household using the information in the database. " \
"""This is the database schema:

```

/* Table of object names in the household with their corresponding concept name */
CREATE TABLE objects
(
    name            varchar(50) NOT NULL,
    concept_name    varchar(50) NOT NULL,
    PRIMARY KEY (name)
);

CREATE TYPE attribute_type as ENUM ('other_object', 'bool', 'int', 'float', 'str');

/* Table of the attributes that objects can have, including the name of the attribute and its data type */
CREATE TABLE attribute_names_and_types
(
    attribute_name varchar(50)    NOT NULL,
    type           attribute_type NOT NULL,
    PRIMARY KEY (attribute_name)
);

/* A table of all the attributes objects have where the attribute is of type string */
CREATE TABLE object_attributes_str
(
    name            varchar(50) NOT NULL,
    attribute_name  varchar(50) NOT NULL,
    attribute_value varchar(50) NOT NULL,
    PRIMARY KEY (name, attribute_name, attribute_value),
    FOREIGN KEY (name)
        REFERENCES objects (name)
        ON DELETE CASCADE,
        ON UPDATE CASCADE,
    FOREIGN KEY (attribute_name)
        REFERENCES attribute_names_and_types (attribute_name)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

/* A table of all the attributes objects have where the attribute is of type integer */
CREATE TABLE object_attributes_int
(
    name            varchar(50) NOT NULL,
    attribute_name  varchar(50) NOT NULL,
    attribute_value int         NOT NULL,
    PRIMARY KEY (name, attribute_name, attribute_value),
    FOREIGN KEY (name)
        REFERENCES objects (name)
        ON DELETE CASCADE,
        ON UPDATE CASCADE,
    FOREIGN KEY (attribute_name)
        REFERENCES attribute_names_and_types (attribute_name)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

/* A table of all the attributes objects have where the attribute is of type float */
CREATE TABLE object_attributes_float
(
    name            varchar(50) NOT NULL,
    attribute_name  varchar(50) NOT NULL,
    attribute_value double precision NOT NULL,
    PRIMARY KEY (name, attribute_name, attribute_value),
    FOREIGN KEY (name)
        REFERENCES objects (name)
        ON DELETE CASCADE,
        ON UPDATE CASCADE,
    FOREIGN KEY (attribute_name)
        REFERENCES attribute_names_and_types (attribute_name)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

/* A table of all the attributes objects have where the attribute is of type boolean */
CREATE TABLE object_attributes_bool
(
    name            varchar(50) NOT NULL,
    attribute_name  varchar(50) NOT NULL,
    attribute_value bool,
    PRIMARY KEY (name, attribute_name, attribute_value),
    FOREIGN KEY (name)
        REFERENCES objects (name)
        ON DELETE CASCADE,
        ON UPDATE CASCADE,
    FOREIGN KEY (attribute_name)
        REFERENCES attribute_names_and_types (attribute_name)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

/* A table of all the attributes objects have where the attribute points to another object */
/* It might be worthwhile to separately investigate attributes associated with the other objects found here*/
CREATE TABLE object_attributes_other_object
(
    name            varchar(50) NOT NULL,
    attribute_name  varchar(50) NOT NULL,
    other_object_name   varchar(50) NOT NULL,
    PRIMARY KEY (name, attribute_name, other_object_name),
    FOREIGN KEY (name)
        REFERENCES objects (name)
        ON DELETE CASCADE,
        ON UPDATE CASCADE,
    FOREIGN KEY (attribute_name)
        REFERENCES attribute_names_and_types (attribute_name)
        ON DELETE CASCADE
        ON UPDATE CASCADE,
    FOREIGN KEY (other_object_name)
        REFERENCES objects (name)
        ON DELETE CASCADE
        ON UPDATE CASCADE
);

```

""" \
"You will be asked a query by the user. " \
"You are allowed to run arbitrarily many SQL queries on the database to gather the information you need to answer the user query. " \
"First determine what the attributes and their types are to find out what information you can query for. " \
"Then determine which objects are relevant to the user query and keep track of their entity_id. " \
"Then collect information about those objects by querying the attribute tables with those entity_id's. " \
"Try to make your queries only match specific entity_ids. "

	functions = [
		{
			"name": "run_sql",
			"description": "Runs a given SQL query and returns the result",
			"parameters": {
				"type": "object",
				"properties": {
					"query": {
						"type": "string",
						"description": "A syntactically correct SQL query",
					},
					"reasoning": {
						"type": "string",
						"description": "A short explanation of how the SQL query will help answer the user query"
					}
				}
			}
		}
	]

	def __init__(self, verbose: bool = False) -> None:
		self.verbose = verbose

	def run_psql_command(self, command: str):
		try:
			# Run the psql command as a subprocess
			psql_process = subprocess.Popen(
				["psql", "postgresql://postgres:password@localhost:5432/knowledge_base", "-c", command, "-P", "pager=off", "-P", "footer=off"],
				stdin=subprocess.PIPE,
				stdout=subprocess.PIPE,
				stderr=subprocess.PIPE,
				text=True  # This ensures that the output is returned as text
			)

			# Wait for the process to complete and fetch the output
			stdout, stderr = psql_process.communicate()

			# Check if there were any errors
			if psql_process.returncode == 0:
				# Successful execution
				return stdout
			else:
				# Error occurred
				return f"Error executing SQL query:\n{stderr}"
		except Exception as e:
			return f"Error executing SQL query: {str(e)}"		

	def input_initial_state(self, initial_state: str, knowledge_yaml: str) -> None:
		with tempfile.NamedTemporaryFile(mode='w') as temp_yaml_file:
			temp_yaml_file.write(knowledge_yaml)
			all_knowledge = [load_knowledge_from_yaml(temp_yaml_file.name)]
		concept_count, instance_count = populate_with_knowledge(get_default_ltmc(), all_knowledge)
		if self.verbose:
			print("Loaded {} concepts and {} instances".format(concept_count, instance_count))
		self.objects = self.run_psql_command("SELECT * from objects")
		self.attributes = self.run_psql_command("SELECT * FROM attribute_names_and_types")
		if self.verbose:
			print(self.objects)
			print(self.attributes)
	
	def input_state_change(self, state_change: str) -> None:
		pass

	def answer_query(self, query: str) -> str:
		messages = [
			{
				"role": "system",
				"content": SQLAgent.initial_prompt
			},
			{
				"role": "user",
				"content": query
			},
			{
				"role": "assistant",
				"content": None,
				"function_call": {
					"name": "run_sql",
					"arguments": \
"""{
	"query": "SELECT * from objects",
	"reasoning": "I want to know the names and associated concepts of all the objects to find out which ones are relevant to the user query"
}"""
				}
			},
			{
				"role": "function",
				"name": "run_sql",
				"content": self.objects
			},
			{
				"role": "assistant",
				"content": None,
				"function_call": {
					"name": "run_sql",
					"arguments": \
"""{
	"query": "SELECT * FROM attribute_names_and_types",
	"reasoning": "I want to know what different attributes objects can have"
}"""
				}
			},
			{
				"role": "function",
				"name": "run_sql",
				"content": self.attributes
			}
		]

		while True:
			response = openai.ChatCompletion.create(
				model="gpt-3.5-turbo-0613",
				messages=messages,
				functions=SQLAgent.functions,
				function_call="auto",
				temperature=0
			)
			message = response["choices"][0]["message"]
			messages.append(message)

			if message.get("function_call"):
				if message["function_call"]["name"] != "run_sql":
					raise ValueError("Invalid function call: " + str(message))
				args = json.loads(message["function_call"]["arguments"])
				if not args.get("query"):
					raise ValueError("Did not pass in 'query' parameter")
				sql_query = args["query"]
				if not isinstance(sql_query, str):
					raise ValueError("Invalid type for 'query' argument")
				response = self.run_psql_command(sql_query)
				if self.verbose:
					print("SQL query: ", sql_query)
					print(f"Response:\n", response)
				messages.append({
					"role": "function",
					"name": "run_sql",
					"content": response
				})
				if self.verbose:
					print(message)
			else:
				break

		return message["content"]

def main():
	sim = Simulation(Dataset("test"), SQLAgent(verbose=True))
	sim.run()

if __name__ == "__main__":
	main()