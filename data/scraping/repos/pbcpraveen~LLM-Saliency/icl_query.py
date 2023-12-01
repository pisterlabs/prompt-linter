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
import random
import time

path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))
import propmts
from common_utils import *
from constants import *
from utils import *

load_dotenv('../api_key.env')
openai.api_key = os.getenv("OPENAI_API_KEY")

setup_directories()
logger = get_logger(F"{MOVIE}.log", depth="INFO")

with open("../knowledge_validation/dataset/nobel_laureates_attribute_verified.pickle", "rb") as file:
    data = pickle.load(file)
target_attribute = Attribute.WORK_CITY.value
concept_class = None
meta = metadata[NOBEL_LAUREATES]

for i in meta[TARGET_ATTRIBUTES].keys():
    if target_attribute in meta[TARGET_ATTRIBUTES][i]:
        concept_class = i
entities = []
icl_examples = {}
for concept_class in meta[TARGET_ATTRIBUTES].keys():
    icl_examples[concept_class] = []
    for target_attribute in meta[TARGET_ATTRIBUTES][concept_class]:
        entities = []
        idx = 0
        for entry in data:
            if entry[ATTRIBUTE_VERIFIED][target_attribute]:
                e = {}
                e[INDEX_COLUMN] = idx
                for attr in meta[CONTEXTUALISING_ATTRIBUTES]:
                    e[attr] = entry[CONTEXTUALISING_ATTRIBUTES][attr]
                e[target_attribute] = entry[TARGET_ATTRIBUTES][concept_class][target_attribute]
                entities.append(e)
                idx += 1
        icl_attr = get_icl_examples(entities, meta, target_attribute, count=min(500, len(entities)))
        icl_attr[TARGET_ATTRIBUTES] = target_attribute
        icl_examples[concept_class].append(icl_attr)

for concept_class in icl_examples.keys():
    file = open(f"dataset/icl_query_{concept_class}_{meta[ENTITY]}.pickle", "wb")
    data_concept = pd.concat(icl_examples[concept_class])
    prompts = data_concept[ICL_PROMPT_COLUMN].to_list()
    if concept_class == ConceptClass.YEAR.value:
        ground_truth = list(map(extract_year, data_concept[GROUND_TRUTH].to_list()))
    else:
        ground_truth = data_concept[GROUND_TRUTH].to_list()
    target_attr = data_concept[TARGET_ATTRIBUTES].to_list()

    responses = create_and_run_api_request_threads(prompts, 5, logger)
    responses = list(itertools.chain(*responses))
    df = pd.DataFrame()
    df[ICL_PROMPT_COLUMN] = prompts
    df[GROUND_TRUTH] = ground_truth
    df[TARGET_ATTRIBUTES] = target_attr
    df[GPT_4_RESPONSE] = responses

    pickle.dump(df, file)
    file.close()
    logger.info(f"Finished writing {concept_class} ICL queries to file")


