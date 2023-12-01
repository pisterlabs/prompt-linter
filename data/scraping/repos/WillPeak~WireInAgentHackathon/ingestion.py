
from dotenv import load_dotenv
import os
import weaviate
from dateutil.parser import parse
from dateutil.tz import tzutc
import openai
load_dotenv()


WEAVIATE_URL = os.getenv('WEAVIATE_URL')
WEAVIATE_API_KEY = os.getenv('WEAVIATE_API_KEY')


# Define a function to get the embedding
def get_embedding(text):
    return  openai.Embedding.create(input=text, model="text-embedding-ada-002")["data"][0]["embedding"]


client = weaviate.Client(url=WEAVIATE_URL, auth_client_secret=weaviate.AuthApiKey(WEAVIATE_API_KEY))

# Define schema for Chat
expected_schema = {
    "classes": [
        {
            "class": "Chat",
            "properties": [
                {"name": "username", "dataType": ["string"]},
                {"name": "content", "dataType": ["string"]},
                {"name": "timestamp", "dataType": ["date"]},
                {"name": "role", "dataType": ["string"]},
                        {
          "name": "vector",
          "dataType": ["number[]"]
        },
            ]
        }
    ]
}


# Get the classes from Weaviate
classes = client.schema.get()['classes']

# Check if the 'Chat' class exists
if not any(class_obj.get('class') == 'Chat' for class_obj in classes):
    # Create the schema in Weaviate if 'Chat' does not exist
    client.schema.create(expected_schema)

def ingest_user_chat(username, messages, timestamp):
    timestamp_parsed = parse(timestamp)
    timestamp_rfc3339 = timestamp_parsed.replace(tzinfo=tzutc()).isoformat()
    for message in messages:
        vector_values = get_embedding(message['content'])
        # Create the chat object
        client.data_object.create({
            "username": username,
            "content": message['content'],
            "timestamp": timestamp_rfc3339,
            "vector": vector_values,
            "role": message['role']
        }, "Chat")
        print("Ingestion Complete")