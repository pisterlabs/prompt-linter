import os
from dotenv import load_dotenv
load_dotenv()
import argparse
from pymongo import MongoClient
from pymongo import InsertOne, DeleteMany, ReplaceOne, UpdateOne
import pandas as pd
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_columns', None)

import openai
openai.api_key = os.environ["OPENAI_API_KEY"]

def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']

def add_embedding(collection):
    '''
    meet requirements
        (1) length >100
        (2) bertopic exit and != 'Unknown'
        (3) not be trained by last time

    can adjust batch size to control parallel

    Args:
        collection:

    Returns:

    '''

    import bson

    batch_size = 1000
    message_len_threshold = 100

    selection_criteria = {
        "topicUpdateDate": {'$exists': False},
        "predicted_class": {'$exists': True, "$ne": 'Unknown'},
        "$expr": {"$gt": [{"$strLenCP": '$messageText'}, message_len_threshold]},
    }
    projection = {'_id': 1, 'messageText': 1}
    cursor = collection.find_raw_batches(selection_criteria, projection, batch_size=batch_size)

    # Iterate through the cursor in batches
    for batch in cursor:
        data = bson.decode_all(batch)
        df = pd.DataFrame(list(data))

        print("messages meet requirements: ", len(df))
        df['embedding'] = df['messageText'].apply(lambda x: get_embedding(x))

        tmp = list()
        for index, row in df.iterrows():
            tmp.append(UpdateOne({"_id": row["_id"]}, {"$set": {"embedding": row['embedding']}}))
        collection.bulk_write(tmp)

if __name__ == '__main__':

    '''
    Add messageDate to the whole collection: scrape.telegram
    use command:
        (1) prd dataset
        python src/pipeline/3_assignEmbeddingToMessage.py -o scrape.telegram

        (2) testing dataset
        python src/pipeline/3_assignEmbeddingToMessage.py -o test.telegram
    
    '''

    # parse parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output_database', help="Specify the output database", required=True)
    args = parser.parse_args()

    # connect to db
    ATLAS_TOKEN = os.environ["ATLAS_TOKEN"]
    ATLAS_USER = os.environ["ATLAS_USER"]
    cluster = MongoClient(
        "mongodb+srv://{}:{}@cluster0.fcobsyq.mongodb.net/".format(ATLAS_USER, ATLAS_TOKEN))
    db_name, collection_name = args.output_database.split('.')
    collection = cluster[db_name][collection_name]

    # update embedding for new coming data
    add_embedding(collection)

    cluster.close()