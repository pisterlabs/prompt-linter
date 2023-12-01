import itertools
import sys
import os
from pathlib import Path
import pickle
import pandas as pd
import json
import openai
import pickle
import dotenv

path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))
import propmts
from common_utils import *
from constants import *
from utils import *


def get_prompt(row):
    return propmts.NOBLE_LAUREATES_VALIDATION_QUERY.format(row[Attribute.YEAR.value],
                                                           row[Attribute.CATEGORY_NOBEL.value])


def format_name(record):
    first_name = record[CONTEXTUALISING_ATTRIBUTES][Attribute.FIRST_NAME.value]
    last_name = record[CONTEXTUALISING_ATTRIBUTES][Attribute.SURNAME.value]
    name = ''
    if first_name is not None:
        name += first_name
    if last_name is not None:
        name += (' ' + last_name)
    return name


load_dotenv('../api_key.env')
openai.api_key = os.getenv("OPENAI_API_KEY")
setup_directories()
logger = get_logger('noble_laureates.log')

logger.info("Checking if LLM has been already queried")
if "noble_laureates_LLM_recall.csv" not in os.listdir('dataset/'):
    logger.info("Querying LLM for knowledge validation")
    data = pickle.load(open("../data/dataset/noble_prize.pickle", "rb"))

    record = set()
    for i in data:
        record.add((i[CONTEXTUALISING_ATTRIBUTES][Attribute.CATEGORY_NOBEL.value],
                    i[TARGET_ATTRIBUTES][ConceptClass.YEAR.value][Attribute.YEAR.value]))

    df = pd.DataFrame()
    df[Attribute.CATEGORY_NOBEL.value] = [i[0] for i in record]
    df[Attribute.YEAR.value] = [i[1] for i in record]


    df[PROMPT_COLUMN] = df.apply(get_prompt, axis = 1)
    prompts = df.prompt.to_list()


    responses = create_and_run_api_request_threads(prompts, 5, logger)
    responses = list(itertools.chain(*responses))

    df[GPT_4_RESPONSE] = responses
    logger.info('Saving responses to ./dataset/noble_laureates_LLM_recall.csv')
    df.to_csv("dataset/noble_laureates_LLM_recall.csv")

logger.info("Reading LLM responses from ./dataset/noble_laureates_LLM_recall.csv")
df = pd.read_csv("dataset/noble_laureates_LLM_recall.csv")
lookup = dict()
data = df.T.to_dict().values()

logger.info("Creating a lookup dictionary with LLM response")
for i in data:
    if i[Attribute.YEAR.value] not in lookup.keys():
        lookup[i[Attribute.YEAR.value]] = dict()
    lookup[i[Attribute.YEAR.value]][i[Attribute.CATEGORY_NOBEL.value]] = get_names(i[GPT_4_RESPONSE])

logger.info("Reading prepared dataset from ../data/dataset/nobel_prize.pickle")
data = pickle.load(open("../data/dataset/noble_prize.pickle", "rb"))

logger.info("Creating verification flag")
for i in range(len(data)):
    record = data[i]
    name = format_name(record)
    category = record[CONTEXTUALISING_ATTRIBUTES][Attribute.CATEGORY_NOBEL.value]
    year = int(record[TARGET_ATTRIBUTES][ConceptClass.YEAR.value][Attribute.YEAR.value])
    flag = name_match(candidate=name, responses=lookup[year][category])
    data[i][VERIFIED] = flag

logger.info("Writing data with verified flag to dataset/noble_prize_verified.pickle")
with open(f'dataset/{VERIFIED_RECORDS[NOBEL_PRIZE]}', 'wb') as handle:
    pickle.dump(data, handle)
