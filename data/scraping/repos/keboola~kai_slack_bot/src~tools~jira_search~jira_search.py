import logging
from atlassian import Jira
from llama_index import SimpleDirectoryReader, GPTVectorStoreIndex, VectorStoreIndex
from langchain.llms import OpenAI
from llama_index import Document
import os
import sys
from decouple import config
from llama_index.node_parser import SimpleNodeParser




jira = Jira(
    url='https://keboola.atlassian.net',
    username=config("CONFLUENCE_USERNAME"),
    password=config("CONFLUENCE_PASSWORD")
)

pre_prompt = """
You are being given a question that is meant to be answered by searching the
Keboola Jira. The only way to answer this question is to search the Jira.
In order to search the Jira, you must generate a JQL query that will
return the most relevant results.

Jira uses JQL for text indexing,
which provides a rich query language.
Much of the information about how to generate a JQL query
is derived from the Query Parser Syntax
page of the Lucene documentation.

A query is broken up into terms and operators.
There are two types of terms: Single Terms and Phrases.

A Single Term is a single word such as "test" or "hello".

A Phrase is a group of words surrounded by double quotes such as "hello dolly".

Note: All query terms in Jira are case insensitive.

Your task is to take the input question and generate a JQL query that will return the most relevant results.

Examples:
text ~ "Keboola AND Python AND component"
text ~ "BYODB AND process"
"""


def generate_jql_query_keywords(input_text: str) -> str:
    llm = OpenAI()
    response = llm.predict(pre_prompt + input_text)
    return response

def validate_jql_query(query: str) -> bool:
    try:
        jira.jql(query)
        return True
    except:
        return False

# TODO: Have this return an array of documents that will subsequently be turned into an index
def jira_search(input_text: str) -> Document:
    """
    Search the Keboola Jira for the most relevant results to the input text.
    
    Args:
        input_text: Input question or text from which to extract keywords.
        
    Returns:
        A vector store index containing the most relevant results to the input text.
    """
    
    if not input_text:
        logging.error('Input text is empty.')
        raise ValueError('Input text cannot be empty.')
    
    try:
        llm = OpenAI()
        jql_query = llm.predict(pre_prompt + input_text)
    except Exception as e:
        logging.error(f'Failed to generate JQL query: {e}')
        raise
    
    return jira.jql(jql_query)



def construct_nodes(documents):
    #TODO: Explore more advanced node parsing/construction
    parser = SimpleNodeParser()
    nodes = parser.get_nodes_from_documents(documents)
    return nodes

def construct_index(nodes):
    index = VectorStoreIndex(nodes)
    return index


def query_engine(query, index):
    return index.as_query_engine().query(query)