# add_disposition.py
# For each document in the conversation container, generate a disposition
# change document and write it to the disposition container.

# Requirements:
# pip install azure-cosmos
# pip install tqdm
# pip install openai
#
# os.environ["COSMOS_CONNECTION_STRING"] - set to your cosmosdb connection string
# os.environ["OPENAI_API_KEY"] - set to your openai api key

import os
from azure.cosmos import CosmosClient
from tqdm import tqdm
import openai
import re
import uuid

COSMOS_CONNECTION_STRING = os.environ['COSMOS_CONNECTION_STRING']
COSMOS_DATABASE_NAME = 'openmw_conv'

OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
OPENAI_MODEL_NAME = 'gpt-3.5-turbo'
OPENAI_MODEL_TEMPERATURE = 0.85

FORCE_UPDATE_ALL = False

# Input:
json_input_container = 'js_input'
json_output_container = 'js_output'
api_output_continer = 'api_output'

# Output:
disposition_container = 'disposition'

def get_collection(collection_name):
    return db.get_container_client(collection_name)

def get_documents(collection):
    return collection.query_items(
        query='SELECT * FROM c',
        enable_cross_partition_query=True
    )

def get_disposition_change_document_dict(json_input, json_output, api_output):
    # Construct the messages sent to the chat completion api
    messages = json_output['messages'].copy()

    # First, replace the final message with the player's original prompt
    # Replacing the message containing the actor's current disposition
    messages[-1]["content"] = json_input['prompt']

    # Add the model's response to the dialogue
    messages.append({"role": "assistant", "content": api_output['choices'][0]['message']['content']})

    # Add a new message asking the model for a disposition change
    actor_name = json_input['actor']
    player_name = json_input['player_name']
    disp_message = f'''[PAUSE DIALOGUE]
Given {actor_name}'s response to what {player_name} said, how has {actor_name}'s disposition towards {player_name} changed?

Think things through from the perspective of {actor_name}. First, write one or more sentences describing what {actor_name} might be thinking about {player_name}, if anything. Then end your response with [a number between square brackets].

The number should be between -100 and +100, where a positive number indicates a more positive attitude towards {player_name}, and a negative number indicates a more negative attitude towards {player_name}.

For an example of scale:
  * A disposition change of 0 is most common, and is suitable for most small-talk or generic conversation.
  * A disposition change of +/- 5 would be appropriate for a rude or insulting comment, or kind or flattering comment.
  * A change of 20 would be appropriate for larger gestures such as a gift or personal threat.
  * A change of 50 would be appropriate for a major betrayal or a major act of kindness.
  * Changes larger than 50 should be used only in extremely rare circumstances, left open to your discretion.

Feel free to use any number between the examples, depending on how strongly {actor_name} feels.'''
    messages.append({"role": "user", "content": disp_message})

    response = openai.ChatCompletion.create(
        model = OPENAI_MODEL_NAME,
        temperature = OPENAI_MODEL_TEMPERATURE,
        messages = messages,
    )
    response_text = response.choices[0]['message']['content']

    try:
        # Look for the disposition change, the number between square brackets
        # Use a regular expression to look for "[(+/-)number]"
        # Make sure to allow for 3 variations: "[+12]" "[12]" and "[-12]" as the model is unpredictable.
        re_expression = r'\[([+-]?\d+)\]'
        re_search = re.search(re_expression, response_text)

        # Get the number between the square brackets and convert to an int
        disposition_change = int(re_search.group(1))

        # Construct the disposition change document
        message_id = json_input['message_id']
        disposition_change_document = {
            'id': message_id,
            'disposition_change': disposition_change,
            'model': OPENAI_MODEL_NAME,
            'temperature': OPENAI_MODEL_TEMPERATURE,
            'messages': messages,
            'response': response,
            'message_id': message_id,
            'document_id': str(uuid.uuid4()),
        }

        return disposition_change_document
    except Exception as e:
        print(f'Error parsing disposition change for document {json_input["id"]}')
        print(e)
        return None

client = CosmosClient.from_connection_string(COSMOS_CONNECTION_STRING)
db = client.get_database_client(COSMOS_DATABASE_NAME)

json_input_documents_list = list(get_documents(get_collection(json_input_container)))
json_input_documents_dict = {doc['id']: doc for doc in json_input_documents_list}

json_output_documents_list = list(get_documents(get_collection(json_output_container)))
json_output_documents_dict = {doc['id']: doc for doc in json_output_documents_list}

api_output_documents_list = list(get_documents(get_collection(api_output_continer)))
api_output_documents_dict = {doc['id']: doc for doc in api_output_documents_list}

disposition_documents_list = list(get_documents(get_collection(disposition_container)))
disposition_documents_dict = {doc['id']: doc for doc in disposition_documents_list}

# Sanity check - Make sure there's the same number of documents in all 3 input containers
# TODO: If this fails, maybe just fallback to the intersection of the three sets?
assert len(json_input_documents_dict) == len(json_output_documents_dict) == len(api_output_documents_dict)

documents_needing_updates = set(json_input_documents_dict.keys()) - set(disposition_documents_dict.keys())

if FORCE_UPDATE_ALL:
    documents_needing_updates = set(json_input_documents_dict.keys())

print(f'Found {len(documents_needing_updates)} documents needing updates')

for document_id in tqdm(documents_needing_updates):
    json_input = json_input_documents_dict[document_id]
    json_output = json_output_documents_dict[document_id]
    api_output = api_output_documents_dict[document_id]

    disposition_change_document = get_disposition_change_document_dict(json_input, json_output, api_output)

    # Debug: print and break
    #print(disposition_change_document)
    #break

    if disposition_change_document is None:
        continue
    
    # Write the disposition change document to the disposition container
    disposition_collection = get_collection(disposition_container)
    disposition_collection.upsert_item(disposition_change_document)