import logging
import os
import sys

from . import extract_queries_from_result
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain import PromptTemplate, LLMChain
import requests
from requests.auth import HTTPBasicAuth

REPAIR_TEMPLATE = os.environ.get('REPAIR_PROMPT', """
You are an smart SQL assistant, capable of fixing SQL queries based on instructions and feedback. Follow that guidelines:
- write a syntactically correct query using {dialect}
- do not fancy format the query with newlines or tabs, just return the raw query
- if you want to query multiple tables, use the JOIN keyword or subselects to join the tables together
- use tables and columns from the provided schema to generate the valid result from the provided hints
- always generate valid queries including columns and tables from the schema
- if there are repeated columns, prefix column names with table names
- do not include any comments in the query, just the query itself
- generate queries with real examples, not using placeholders
- end your query with a semicolon
- only show the totally completed final query, without any additional output
- generate only one query
- if you cannot fix the query, return an empty string
- you only can use tables and columns defined in this schema:

{table_info}

The initial query was:

{query}

It gave the error:

{error_message}

Please suggest a valid query based on the previous one.
""")

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger.setLevel(logging.INFO)

def predict_repair(llm, query, error_message, docsearch, dialect):    
    # different search types
    if (os.environ.get('SEARCH_TYPE', 'similarity') == 'mmr'):
        docs = docsearch.max_marginal_relevance_search(
            query, k=int(os.environ.get('DOCS_TO_RETRIEVE', 5)))
    else:
        docs = docsearch.similarity_search(
            query, k=int(os.environ.get('DOCS_TO_RETRIEVE', 5)))

    #  no queries stored, go with llm
    prompt = PromptTemplate(
        input_variables=["query", "error_message", "table_info", "dialect"], template=REPAIR_TEMPLATE)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    res = llm_chain.predict(table_info=docs, query=query, error_message=error_message, dialect=dialect)
    logger.info("Result from LLM: "+res)
    return res

def repair_chat(query, error_message, docsearch, dialect):
    llm = ChatOpenAI(temperature=os.environ.get('TEMPERATURE', 0.9),
                     model_name=os.environ.get('REPAIR_MODEL', 'gpt-3.5-turbo'), n=int(os.environ.get('OPENAI_NUM_ANSWERS', 1)))
    res = predict_repair(llm, query, error_message, docsearch, dialect)
    final_queries = extract_queries_from_result(res)
    return final_queries

def repair_selfhosted(query, error_message, docsearch, dialect):
    # different search types
    if (os.environ.get('SEARCH_TYPE', 'similarity') == 'mmr'):
        docs = docsearch.max_marginal_relevance_search(
            query, k=int(os.environ.get('DOCS_TO_RETRIEVE', 5)))
    else:
        docs = docsearch.similarity_search(
            query, k=int(os.environ.get('DOCS_TO_RETRIEVE', 5)))

    prompt = PromptTemplate(
        input_variables=["query", "table_info", "dialect"], template=REPAIR_TEMPLATE)
    query = prompt.format(query=query, error_message=error_message, table_info=docs, dialect=dialect)

    # issue a request to an external API
    request = {
        'prompt': query,
        'temperature': float(os.environ.get('TEMPERATURE', 1.3)),
        'top_p': 0.1,
        'typical_p': 1,
        'repetition_penalty': 1.18,
        'top_k': 40,
        'min_length': 0,
        'no_repeat_ngram_size': 0,
        'num_beams': 1,
        'penalty_alpha': 0,
        'length_penalty': 1,
        'early_stopping': False,
        'seed': -1,
        'add_bos_token': True,
        'truncation_length': 2048,
        'ban_eos_token': False,
        'skip_special_tokens': True,
        'stopping_strings': []
    }

    try:
        response = requests.post(os.environ.get('LLM_HOST', ''), json=request, auth=HTTPBasicAuth(
            os.environ.get('LLM_USER', ''), os.environ.get('LLM_PASSWORD', '')))
        if response.status_code == 200:
            result = response.json()['results'][0]['text']
            logger.info("Result from LLM: "+result)
            final_queries = extract_queries_from_result(result)
            return final_queries
    except Exception as e:
        logger.exception("Error in autocomplete_selfhosted: "+e)

    return None

def repair_query_suggestions(query, error_message, docsearch, dialect):
    if os.environ.get('REPAIR_METHOD', 'chat') == 'chat':
        queries = repair_chat(query, error_message, docsearch, dialect)
    elif os.environ.get('REPAIR_METHOD', 'chat') == 'selfhosted':
        queries = repair_selfhosted(query, error_message, docsearch, dialect)
    else:
        queries = repair_chat(query, error_message, docsearch, dialect)
    logger.info("Returned queries are: ")
    logger.info(queries)
    return queries
