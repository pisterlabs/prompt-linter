import os
import re
import openai
from dotenv import load_dotenv
from src.mongo.pymongo_get_database import get_database

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.organization = os.getenv("ORGANIZATION_KEY")

dbname = get_database()
collection_name = dbname["MicrosoftNews"]

for item in collection_name.find():
    if "sentiment" in item and type(item["sentiment"]) == str:
        match = re.search(r'[$]{0}[-+]?[0-9]*\.?[0-9]+[%]{0}', item["sentiment"])

        if match == None:
            collection_name.delete_one({"_id" : item["_id"]})
        else:
            myQuery = {"_id" : item["_id"]}
            newValues = {"$set" : {"sentiment" : float(match.group(0))}}
            collection_name.update_one(myQuery, newValues)

        continue
    elif "sentiment" in item and type(item["sentiment"]) == float:
        continue
    
    try:
        promptText = "I want you to analyze the next news article I give you, extract a sentiment score from it, and evaluate how positive or negative it is for the company Microsoft. '-10' being extremely negative and '10' being extremely positive. Don't forget to consider relevancy. You are allowed to use floating-point numbers. Don't explain anything further. Don't use any other character. Only '+', '-', '.' and numbers."
        promptText += item["content"]
        promptText = promptText[:16388]

        completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": promptText}], top_p=0.1)

        myQuery = {"_id" : item["_id"]}
        newValues = {"$set" : {"sentiment" : completion["choices"][0]["message"]["content"]}}
        collection_name.update_one(myQuery, newValues)

    except:
        print("An error occured")
