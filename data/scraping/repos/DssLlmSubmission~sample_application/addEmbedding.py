import os
from dotenv import load_dotenv
load_dotenv()
from pymongo import MongoClient
import pandas as pd
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

import openai
openai.api_key = os.environ["OPENAI_API_KEY"]


# useful links for fetching data from MongoDB
# https://www.w3schools.com/python/python_mongodb_query.asp
# https://www.mongodb.com/docs/manual/reference/method/db.collection.find/

# connect to db
ATLAS_TOKEN = os.environ["ATLAS_TOKEN"]
ATLAS_USER = os.environ["ATLAS_USER"]
cluster = MongoClient(
    "mongodb+srv://{}:{}@cluster0.fcobsyq.mongodb.net/".format(ATLAS_USER, ATLAS_TOKEN))

# specify names of database and collection
# db_name, collection_name = "test", "telegram_sample"
db_name, collection_name = "scrape", "telegram"
collection = cluster[db_name][collection_name]

def get_chats_list(input_file_path):
    """
    Args:
        input_file_path: chats path

    Returns: pandas dataframe. e.g.
            |country|chat|
            |Switzerland|https://t.me/zurich_hb_help|
            |Switzerland|https://t.me/helpfulinfoforua|
    """
    countries, chats = list(), list()
    with open(input_file_path, 'r') as file:
        for line in file.readlines():
            if line.startswith("#"):
                country = line.replace('#', '').replace('\n', '')
            else:
                chat = line.replace('\n', '')

                chats.append(chat)
                countries.append(country)

    df = pd.DataFrame(list(zip(countries, chats)),
                      columns=['country', 'chat'])
    return df

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

if __name__ == '__main__':
    chats = get_chats_list('../../../../data/telegram/queries/DACH.txt')
    print(chats)

    for index, row in chats.iterrows():
        condition = {'chat': row['chat']}
        selection = {'_id': 1, 'messageText': 1, 'embedding':1 }
        query_res = collection.find(condition, selection)  # use find, find_one to perform query
        print(row['chat'])
        for i in query_res:
            if 'embedding' not in i and len(i['messageText']) > 40:
                print(i)
                embedding = get_embedding(i['messageText'])
                collection.update_one({"_id": i["_id"]}, {"$set": {"embedding":embedding}})

            # if 'embedding' in i and len(i['messageText']) < 40:
            #     print(i)
            #     collection.update_one({"_id": i["_id"]}, {'$unset': {'embedding':1}}) # remove field