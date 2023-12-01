import os
import sys
import time
import pytz
import json
from datetime import datetime
from dotenv import load_dotenv
import pymongo

from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv('OPEN_KEY')

sys.path.append('../Credit_All_In_One/')
import my_logger
from my_configuration import _get_mongodb, _get_pgsql, _get_redis


persist_directory = './chroma_db'
embedding = OpenAIEmbeddings() # default: "text-davinci-003", try to find replacable embedding function



# datetime
taiwanTz = pytz.timezone("Asia/Taipei") 
now = datetime.now(taiwanTz)
today_date = now.date()
today = now.strftime('%Y-%m-%d')

# create a logger
dev_logger = my_logger.MyLogger('etl')
dev_logger.console_handler()
dev_logger.file_handler(today)


def count_mongodb_docs():
    mongo_db = _get_mongodb()
    mongo_collection = mongo_db["official_website"]
    return mongo_collection.count_documents({})
    

def get_chroma_content():
    vectordb = Chroma(persist_directory=persist_directory)
    # dev_logger.info(f'keys: {vectordb.get().keys()}')
    dev_logger.info(f'num of split contents:{len(vectordb.get()["ids"])}') 
    # print(vectordb.get(include=["embeddings","documents", "metadatas"])) 


def truncate_chroma():
    vectordb = Chroma(persist_directory=persist_directory)
    vectordb.delete_collection()
    vectordb.persist()
    vectordb = None
    dev_logger.info('Truncate chromaDB collection.')


def fetch_latest_from_mongodb(logger, pipeline, collection:str, projection:dict, *args, **kwargs) -> list:
    mongo_db = _get_mongodb()
    mongo_collection = mongo_db[collection]
    max_create_dt = mongo_collection.find_one(sort=[('create_dt', pymongo.DESCENDING)])['create_dt']
    # print(f'max_create_dt:{max_create_dt}')
    searching = {'create_dt':max_create_dt}
    if kwargs:
        if kwargs['push']:
            searching['push'] = kwargs['push']
        if kwargs['sorting']:
            cursor = mongo_collection.find(searching, projection).sort(kwargs['sorting'], pymongo.DESCENDING)
            data = list(cursor)
    else:
        data = list(mongo_collection.find(searching, projection))
        
    if data:
        logger.info(json.dumps({'msg':f'Finish retrieving {pipeline} on {max_create_dt} updated documents.'}))
    return data


def insert_into_redis(logger, pipeline:str, redis_key:str, redis_value:dict, max_retries:int = 5, delay:int = 2):
    redis_conn = _get_redis()
    for trying in range(1, max_retries + 1):
        try:
            redis_conn.set(redis_key, json.dumps(redis_value))
            logger.info(json.dumps({'msg':f'Finish inserting {pipeline} into Redis'}))
            break
        except Exception as e:
            logger.warning(
                json.dumps({'msg':
                    f"Failed to set value of {pipeline} in Redis: {e}"
                    f"Attempt {trying + 1} of {max_retries}. Retrying in {delay} seconds."})
            )
            if trying == max_retries:
                logger.error(json.dumps({'msg':f"Failed to set value of {pipeline} in {max_retries} attempts"}))
            time.sleep(delay)


def fetch_distinct_card_name():
    pg_db = _get_pgsql()
    cursor = pg_db.cursor()

    cursor.execute("SELECT card_name FROM card_dictionary ORDER BY card_name;")
    data = cursor.fetchall()
    data = [i[0] for i in data]
    cursor.close()
    pg_db.close()
    return data


def fetch_distinct_card_alias_name():
    pg_db = _get_pgsql()
    cursor = pg_db.cursor()

    sql = """
        SELECT card_name, \
            ARRAY(SELECT DISTINCT e FROM unnest(ARRAY[REPLACE(UPPER(card_name),' ','')] || \
            ARRAY[REPLACE(LOWER(card_name),' ','')] || string_to_array(card_alias_name,', ') || \
            string_to_array(UPPER(card_alias_name),', ') || string_to_array(LOWER(card_alias_name),', ')) AS e) \
            AS all_card_names
        FROM card_dictionary
        ORDER BY card_name;
        """
    cursor.execute(sql)
    data = cursor.fetchall()
    cursor.close()
    pg_db.close()
    return data


if __name__ == '__main__':
    projection = {'post_title': 1, 'post_author':1 , 'push':1, 'post_dt':1, 'post_link':1, 'article': 1, '_id': 0}
    data = fetch_latest_from_mongodb(logger=dev_logger, pipeline='123', collection="ptt", projection=projection, push={'$gte': 90}, sorting='post_dt')
    print(data)
    # get_chroma_content()