from telethon import TelegramClient, events, sync, errors
from telethon.sessions import StringSession
import asyncio
import os
from dotenv import load_dotenv

load_dotenv()
import datetime
from tqdm import tqdm
import argparse
from geosky import geo_plug

from pymongo import MongoClient, errors
import pandas as pd
import openai

openai.api_key = os.environ["OPENAI_API_KEY"]

# To run this code. You must get your own api_id and
# api_hash from https://my.telegram.org, under API Development.

# if you run code locally, you should store variables in .env file.
# If you run code in Github platform, code will fetch secret/variables automatically
TELEGRAM_API_ID = os.environ["TELEGRAM_API_ID"]
TELEGRAM_API_HASH = os.environ["TELEGRAM_API_HASH"]
TELEGRAM_STRING_TOKEN = os.environ["TELEGRAM_STRING_TOKEN"]
ATLAS_TOKEN = os.environ["ATLAS_TOKEN"]
ATLAS_USER = os.environ["ATLAS_USER"]


def validate_local_file(f):  # function to check if file exists
    if not os.path.exists(f):
        raise argparse.ArgumentTypeError("{0} does not exist".format(f))
    return f


def validate_database(s):
    database_name, collection_name = s.split('.')
    cluster = MongoClient("mongodb+srv://{}:{}@cluster0.fcobsyq.mongodb.net/".format(ATLAS_USER, ATLAS_TOKEN))
    db = cluster[database_name]
    list_of_collections = db.list_collection_names()
    if collection_name not in list_of_collections:
        raise Exception("Collection does not exit")
    return s


def initialize_database(database_name, collection_name):
    '''
    use names of database and collection to fetch specific collection
    Args:
        database_name:
        collection_name:

    Returns:

    '''
    cluster = MongoClient("mongodb+srv://{}:{}@cluster0.fcobsyq.mongodb.net/".format(ATLAS_USER, ATLAS_TOKEN))
    collection = cluster[database_name][collection_name]
    return collection


def get_country_to_state_dict():
    '''
    prepare country-state mapping
    Returns:
            a mapping with country as key and states as value
            e.g. {'Switzerland':['Zurich', 'Zug', 'Vaud', 'Saint Gallen'...], 'Germany':[]}
    '''

    data_state = geo_plug.all_Country_StateNames()
    data_state = data_state.replace('null', ' ')
    res = eval(data_state)

    mapping_state = {}
    for element in res:
        for k, v in element.items():
            mapping_state[k] = v

    mapping_state["Switzerland"].remove("Basel-City")
    mapping_state["Switzerland"].append("Basel")

    return mapping_state


def get_state_to_city_dict():
    '''
    prepare state-city mapping
    Returns:
            a mapping with states as key and city as value
            e.g. {'Zurich':["Winterthur", "Uster", ...], 'Basel':[]}
    '''

    data_city = geo_plug.all_State_CityNames()
    data_city = data_city.replace('null', ' ')
    res = eval(data_city)

    mapping_city = {}
    for element in res:
        for k, v in element.items():
            mapping_city[k] = v

    mapping_city['North Rhine-Westphalia'].append('Cologne')
    mapping_city['Bavaria'].append('Nuremberg')
    mapping_city['Basel'] = mapping_city.pop('Basel-City')

    return mapping_city


def special_translate_chat(chat):
    '''
    In same chats, they are writen in German or French. This functions will standardize their spelling
    Args:
        chat: original chat (string)

    Returns: chat with standard format

    '''
    return chat.replace("Lousanne", "Lausanne") \
        .replace("BielBienne", "Biel/Bienne") \
        .replace("Geneve", "Geneva") \
        .replace("StGallen", "Saint Gallen")


def parse_state_city(chat, country):
    mapping_state = get_country_to_state_dict()
    mapping_city = get_state_to_city_dict()
    chat_standard = special_translate_chat(chat)

    # parse state and city
    chat_states = mapping_state[country]
    state, city = '', ''
    for s in chat_states:
        chat_city = mapping_city[s]
        for c in chat_city:
            if c.upper() in chat_standard.upper():
                city = c
                state = s
                break

            if s.upper() in chat_standard.upper():
                state = s
    return state, city


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)['data'][0]['embedding']


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


async def callAPI(input_file_path):
    """
    This function takes an input file, output folder path
    It reads the input file, extracts the chats and then uses the TelegramClient to scrape message.text and message.date from each chat.
    Appending the chat's URL, message text, and message datetime to different lists.
    Then it creates a dataframe from the lists and saves the dataframe to a CSV file in the specified output folder.

    :input_file_path: .txt file containing the list of chats to scrape, each line should represent one chat
    :output_folder_path: folder path where the output CSV file will be saved containing the scraped data
    """

    data = get_chats_list(input_file_path)

    print(len(data))

    for index, row in tqdm(data.iterrows(), total=data.shape[0]):

        async with TelegramClient(StringSession(TELEGRAM_STRING_TOKEN), TELEGRAM_API_ID, TELEGRAM_API_HASH) as client:

            chat = row['chat']
            country = row['country']
            state, city = parse_state_city(chat, country)

            # find max time in the database
            time_col = 'messageDatetime'  # "update_time"
            search_max_date = output_collection.find_one({"chat": chat}, sort=[(time_col, -1)])
            if search_max_date is None:
                max_time = None
            else:
                # avoid include the record which date is equivalent to max_time_db
                max_time = search_max_date[time_col] + datetime.timedelta(seconds=1)

            print("{} last {} time: {} ".format(chat, time_col, max_time))

            data_list = list()

            async for message in client.iter_messages(chat, reverse=True, offset_date=max_time):

                if message.message is not None and message.message != '':
                    record = dict()
                    record['chat'] = chat
                    record['channel_id'] = message.peer_id.channel_id
                    record['messageDatetime'] = message.date
                    record['update_time'] = datetime.datetime.now()
                    record['country'] = country
                    record['state'] = state
                    record['city'] = city
                    record['messageText'] = message.message
                    record['views'] = message.views if message.views is not None else 0
                    record['forwards'] = message.forwards if message.forwards is not None else 0

                    if len(message.message) > 40:
                        record['embedding'] = get_embedding(message.message)

                    if message.replies is None:
                        record['replies'] = 0
                    else:
                        record['replies'] = message.replies.replies

                    if message.reactions is None:
                        record['reactions'] = []
                    else:
                        reaction = dict()
                        for i in message.reactions.results:
                            try:
                                reaction[i.reaction.emoticon] = i.count
                            except:
                                # same message don't have emotion labels (reaction.emoticon)
                                pass
                        record['reactions'] = reaction

                    data_list.append(record)

            print("data len:{}".format(len(data_list)))

            if len(data_list) > 0:
                output_collection.insert_many(data_list)
            else:
                print("no updated records")


if __name__ == '__main__':
    """
    example usage in command line:

    python src/helper/scraping/telegram_tools/scrapeTelegramChannelMessages.py -i data/telegram/queries/DACH.txt -o scrape.telegram

    Read chats from DACH.txt and store telegram data to database.

    (1) scrape telegram data
    (2) get embedding of each sentence
    (3) parse state and city from chat name

    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_file_path', help="Specify the input file", type=validate_local_file,
                        required=True)
    parser.add_argument('-o', '--output_database', help="Specify the output database", required=True)
    args = parser.parse_args()

    o_database_name, o_collection_name = args.output_database.split('.')
    output_collection = initialize_database(o_database_name, o_collection_name)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(callAPI(args.input_file_path))
    loop.close()
