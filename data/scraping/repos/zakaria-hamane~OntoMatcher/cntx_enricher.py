from langchain.chat_models import ChatOpenAI
from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.utilities import GoogleSerperAPIWrapper
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

import json
import re
import os


class cntxEnricher:
    def __init__(self):
        self.openai_api_key = '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'
        self.llm = ChatOpenAI(openai_api_key=self.openai_api_key, temperature=0.0)
        self.tools = load_tools(
            ["arxiv"],
        )
        self.agent_chain = initialize_agent(
            self.tools,  # Using self.tools instead of tools
            self.llm,  # Using self.llm instead of llm
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
        )

    def split_sentences(self, context):
        if ':' in context:
            context = context.split(': ')[1]
            phrases = re.findall(r'"([^"]*)"', context)
        else:
            phrases = re.findall(r'"([^"]*)"', context)
        return phrases

    def enrich_context(self, entity):
        context = self.agent_chain.run(
            f"give me 1 research paper used {entity} in a sentence",
        )
        phrases = self.split_sentences(context)
        return phrases


def run_arxiv_enricher():
    cntx_enricher = cntxEnricher()
    with open('extracted_data/enriched/all_empty_cntx.json') as json_file:
        entities = json.load(json_file)

    updated_entities = []
    for query_dict in tqdm(entities, total=len(entities), desc="Enriching contexts"):
        query = list(query_dict.keys())[0]
        try:
            cntx = cntx_enricher.enrich_context(query)
            print(f"Enriched {query}")
            query_dict[query] = cntx
        except Exception as e:
            print(f"Could not enrich {query}. Error: {str(e)}")
            query_dict[query] = []
        updated_entities.append(query_dict)

    with open('extracted_data/enriched/all_empty_cntx.json', 'w') as outfile:
        json.dump(updated_entities, outfile, indent=4)


# def run_google_serper_enricher():
#     os.environ["SERPER_API_KEY"] = "40bccac006f35a34e33c58953e1d0ccb38fdb7cd"
#     search = GoogleSerperAPIWrapper()
#
#     with open('extracted_data/enriched/all_empty_cntx.json') as json_file:
#         entities = json.load(json_file)
#
#     updated_entities = []
#     for query_dict in tqdm(entities, total=len(entities), desc="Enriching contexts"):
#         query = list(query_dict.keys())[0]
#         try:
#             results = search.run(f"research papers that mention {query}")
#             list_of_results = results.split(' ... ')
#             print(f"Enriched {query}")
#             query_dict[query] = list_of_results
#         except Exception as e:
#             print(f"Could not enrich {query}. Error: {str(e)}")
#             query_dict[query] = []
#         updated_entities.append(query_dict)
#
#     with open('extracted_data/enriched/all_empty_cntx.json', 'w') as outfile:
#         json.dump(updated_entities, outfile, indent=4)

def enrich_entity_with_google_serper(search, query_dict):
    query = list(query_dict.keys())[0]
    try:
        results = search.run(f"research papers that mentioned {query}")
        list_of_results = results.split(' ... ')
        print(f"Enriched {query}")
        query_dict[query] = list_of_results
    except Exception as e:
        print(f"Could not enrich {query}. Error: {str(e)}")
        query_dict[query] = []
    return query_dict

def run_google_serper_enricher():
    os.environ["SERPER_API_KEY"] = "40bccac006f35a34e33c58953e1d0ccb38fdb7cd"
    search = GoogleSerperAPIWrapper()

    with open('extracted_data/enriched/all_empty_cntx.json') as json_file:
        entities = json.load(json_file)

    updated_entities = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        for query_dict in tqdm(executor.map(enrich_entity_with_google_serper, [search]*len(entities), entities), total=len(entities), desc="Enriching contexts"):
            updated_entities.append(query_dict)

    with open('extracted_data/enriched/all_empty_cntx.json', 'w') as outfile:
        json.dump(updated_entities, outfile, indent=4)


#

if __name__ == "__main__":
    run_google_serper_enricher()
