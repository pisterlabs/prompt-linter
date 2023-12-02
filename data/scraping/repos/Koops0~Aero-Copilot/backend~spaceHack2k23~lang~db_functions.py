import json
from langchain_coher import *
from spaceHack2k23.dynamodb import DynamoDB



# Check if definition exists in DynamoDB
def checkDefinitionExists(file_name, user_id, json_string):
    response = DynamoDB.queryDefinition(file_name, user_id)
    # Parse the JSON string to a Python JSON object
    json_object = json.loads(json_string)
    technical_keywords = set(json_object["technical keywords"])
    technical_phrases = set(json_object["technical phrases"])
    technical_concepts = set(json_object["technical concepts"])

    for item in response["Items"]:
        item_data = item["data"]
        item_dict = json.loads(item_data)

        for key, value in item_dict.items():
            if (
                key in technical_keywords
                or key in technical_phrases
                or key in technical_concepts
            ):
                return value

    return None
