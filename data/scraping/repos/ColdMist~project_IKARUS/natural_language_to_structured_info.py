import numpy as np
from langchain.indexes import GraphIndexCreator
from langchain.llms import OpenAI
from utils.helper_functions import *
import pandas as pd
import argparse
import os

def main(args):

    os.environ["OPENAI_API_KEY"] = args.api_key
    nlp = load_nlp()

    if os.path.isfile(args.text_or_filepath):
        text = read_text_from_file(args.text_or_filepath)
    else:
        text = args.text_or_filepath

    index_creator = GraphIndexCreator(llm=OpenAI(temperature=0))
    graph = index_creator.from_text(text)
    triples = graph.get_triples()
    print(f"detected triples are {triples}")
    store_triples(triples, "data/triples.txt")
    triples = pd.read_table("data/triples.txt", header=None)

    # collect the unique entities
    unique_entities = np.unique(
        (list(list(triples[0].unique()) + list(triples[2].unique())))
    )

    connection_information = obtain_connection_information(nlp, unique_entities)
    #store_to_pickle("data/connection_information.pkl", connection_information)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process a text or a file.')
    parser.add_argument('--text_or_filepath', type=str, help='A text string or a filepath.')
    parser.add_argument('--api_key', type=str, help='OpenAI API key.')
    args = parser.parse_args()
    main(args)
