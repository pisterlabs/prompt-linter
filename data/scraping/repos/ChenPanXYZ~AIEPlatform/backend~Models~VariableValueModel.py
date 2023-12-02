from credentials import *
from Models.VariableModel import VariableModel
import openai

import json
with open('config.json', 'r') as f:
    config = json.load(f)

openai.api_key = config["OPEN_AI_KEY"]


class VariableValueModel:
    # Static methods:
    # Get the latest value for each of the given variables for the given user.
    @staticmethod
    def get_latest_variable_values(variables, user, session = None):
        column_name = 'variable'
        array_list = variables  # Example array list

        # Aggregation pipeline to filter and keep the last occurrence
        pipeline = [
            {
                '$match': {
                    column_name: {'$in': array_list}, 
                    'user': user
                }
            },
            {
                '$sort': {
                    'timestamp': 1       # Sort descending by _id (to get the last occurrence)
                }
            },
            {
                '$group': {
                    '_id': '$' + column_name,
                    'lastDocument': {'$last': '$$ROOT'}
                }
            },
            {
                '$replaceRoot': {'newRoot': '$lastDocument'}
            }
        ]

        # Execute the aggregation pipeline
        result = list(VariableValue.aggregate(pipeline))

        return result
    
    # Insert a new variable value for the given user.
    # TODO: error handling.
    @staticmethod
    def insert_variable_value(variable, session = None):
        try:
            # Need to check if it is a text variable or not.
            theVariable = VariableModel.get_one({'name': variable['variable']}, public = True, session = session)
            if theVariable['type'] == 'text':
                numeric_value = int(VariableValueModel.gpt_reader(theVariable, variable['value']))
                variable['text'] = variable['value']
                variable['value'] = numeric_value
            VariableValue.insert_one(variable, session = session)
            return 200
        except Exception as e:
            print(e)
            return 500
        
    @staticmethod
    def gpt_reader(variable, value):
        messages = []
        messages.append({"role": "system", "content": "You are a teacher."})
        messages.append({"role": "system", "content": "Please return a numeric without any extra information."})
        messages.append({"role": "system", "content": "Please always return a numeric, do not ask follow questions. If you think a text is very bad or irrelevant, feel free to give the min value."})
        messages.append({"role": "system", "content": variable['variableValuePrompt']})
        messages.append({"role": "user", "content": "A text to measure the understanding: " + value})
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo", messages=messages, temperature=0.0
        )
        reply = chat.choices[0].message.content
        return reply