import json
import logging
import sys
import requests
from requests.auth import HTTPBasicAuth
from . import extract_queries_from_result
from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
import os
import logging
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger.setLevel(logging.INFO)

CUSTOM_TEMPLATE = os.environ.get('AUTOCOMPLETE_PROMPT', """
You are an smart SQL assistant, capable of generating SQL queries based on comments or hints. You should generate any queries with the specific guidelines:
- write a syntactically correct query using {dialect}
- start your query with one of the following keywords: SELECT, INSERT, UPDATE, or DELETE, or any other {dialect} valid command
- do not fancy format the query with newlines or tabs, just return the raw query
- if you want to query multiple tables, use the JOIN keyword or subselects to join the tables together
- use tables and columns from the provided schema to generate the valid result from the provided hints
- always return the complete valid SQL query, not just fragments
- always generate valid queries including columns and tables from the schema
- if there are repeated columns, prefix column names with table names
- do not include any comments in the query, just the query itself
- generate queries with real examples, not using placeholders
- end your query with a semicolon
- only show the totally completed final query, without any additional output
- generate only one query
- if you cannot generate the result return an empty string, do not show any other content
- you only can use tables and columns defined in this schema:

{table_info}

For example, a valid query might look like this:

SELECT name, age FROM users WHERE age > 30;

Please generate the complete SQL query based on this hint: {query}

""")

SAMPLE_QUERIES_TEMPLATE = os.environ.get('SAMPLE_QUERIES_PROMPT', """
You are an SQL assistant, capable of generating valid SQL queries. Your goal is to read data from a table and generate relevant example queries that can be used for testing.
The queries need to cover the whole range of SQL language, including SELECT, INSERT, UPDATE and DELETE, using relevant columns from the table.
Use the following premises:
- write a syntactically correct query using {dialect}
- generate queries with real examples, not using placeholders
- end your query with a semicolon

Generate a list of 15 queries based on this table:

{table_info}

The output needs to be just a JSON list with this format:
[
"query1",
"query2",
"query3"
]

Only provide this list without any additional output.
""")

MAX_SIMILARITY_RATIO = os.environ.get('MAX_SIMILARITY_RATIO', 0.55)


def predict(llm, query, docsearch, dialect):    
    # different search types
    if (os.environ.get('SEARCH_TYPE', 'similarity') == 'mmr'):
        docs = docsearch.max_marginal_relevance_search(
            query, k=int(os.environ.get('DOCS_TO_RETRIEVE', 5)))
    else:
        docs = docsearch.similarity_search(
            query, k=int(os.environ.get('DOCS_TO_RETRIEVE', 5)))

    for doc in docs:
        if (doc.metadata['type'] == 'query' and doc.page_content):
            if (doc.page_content.strip().startswith(query.strip())):
                # if it is very similar we return it
                s = SequenceMatcher(None, query, doc.page_content)
                if s.ratio() > MAX_SIMILARITY_RATIO:
                    # very similar, will match
                    return doc.page_content

    #  no queries stored, go with llm
    prompt = PromptTemplate(
        input_variables=["query", "table_info", "dialect"], template=CUSTOM_TEMPLATE)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    res = llm_chain.predict(table_info=docs, query=query, dialect=dialect)
    logger.info("Result from LLM: "+res)
    return res

def autocomplete_chat(query, docsearch, dialect):
    llm = ChatOpenAI(temperature=os.environ.get('TEMPERATURE', 0.9),
                     model_name=os.environ.get('LLM_MODEL', 'gpt-3.5-turbo'), n=int(os.environ.get('OPENAI_NUM_ANSWERS', 1)))
    res = predict(llm, query, docsearch, dialect)
    final_queries = extract_queries_from_result(res)
    return final_queries

def autocomplete_selfhosted(query, docsearch, dialect):
    # different search types
    if (os.environ.get('SEARCH_TYPE', 'similarity') == 'mmr'):
        docs = docsearch.max_marginal_relevance_search(
            query, k=int(os.environ.get('DOCS_TO_RETRIEVE', 5)))
    else:
        docs = docsearch.similarity_search(
            query, k=int(os.environ.get('DOCS_TO_RETRIEVE', 5)))

    prompt = PromptTemplate(
        input_variables=["query", "table_info", "dialect"], template=CUSTOM_TEMPLATE)
    query = prompt.format(query=query, table_info=docs, dialect=dialect)

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


def autocomplete_query_suggestions(query, docsearch, dialect):
    if os.environ.get('AUTOCOMPLETE_METHOD', 'chat') == 'chat':
        queries = autocomplete_chat(query, docsearch, dialect)
    elif os.environ.get('AUTOCOMPLETE_METHOD', 'chat') == 'selfhosted':
        queries = autocomplete_selfhosted(query, docsearch, dialect)
    else:
        queries = autocomplete_chat(query, docsearch, dialect)
    logger.info("Returned queries are: ")
    logger.info(queries)
    return queries


def predict_queries(llm, schema, dialect):
    prompt = PromptTemplate(
        input_variables=["table_info", "dialect"], template=SAMPLE_QUERIES_TEMPLATE)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    res = llm_chain.predict(dialect=dialect, table_info=schema)
    logger.info("Result from LLM: "+res)

    return res


def queries_chat(schema, dialect):
    llm = ChatOpenAI(temperature=os.environ.get('TEMPERATURE', 0.9),
                     model_name=os.environ.get('LLM_QUERIES_MODEL', 'gpt-3.5-turbo'), n=1)
    res = predict_queries(llm, schema, dialect)

    try:
        final_queries = json.loads(res)
    except:
        return []
    return final_queries


def queries_selfhosted(schema, dialect):
    prompt = PromptTemplate(
        input_variables=["table_info", "dialect"], template=SAMPLE_QUERIES_TEMPLATE)
    query = prompt.format(table_info=schema, dialect=dialect)

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
            try:
                final_queries = json.loads(result)
            except:
                return []
            return final_queries
    except Exception as e:
        logger.exception("Error in autocomplete_selfhosted: "+e)

    return None


def generate_queries_for_schema(schema, schema_dict, dialect):
    logger.info("Generating queries for schema: "+schema)
    if os.environ.get('QUERIES_METHOD', 'chat') == 'chat':
        queries = queries_chat(schema, dialect)
    elif os.environ.get('QUERIES_METHOD', 'chat') == 'selfhosted':
        queries = queries_selfhosted(schema, dialect)
    else:
        queries = queries_chat(schema, dialect)

    # validate all queries and return only the accepted ones
    final_queries = []
    for q in queries:
        try:
            final_queries.append(q)
        except Exception as e:
            logger.info("validate_query call failed for query: "+q)
            logger.exception(e)
    return final_queries
