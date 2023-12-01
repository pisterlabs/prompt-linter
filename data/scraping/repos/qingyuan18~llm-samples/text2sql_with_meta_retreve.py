import json
import os
import logging
from collections import Counter
from langchain.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain import PromptTemplate, SagemakerEndpoint
from langchain.chains.question_answering import load_qa_chain
from langchain.embeddings import SagemakerEndpointEmbeddings
from langchain.llms.sagemaker_endpoint import ContentHandlerBase
from langchain.docstore.document import Document
from langchain.memory import ConversationBufferWindowMemory
from langchain import LLMChain
from langchain import PromptTemplate, SagemakerEndpoint
from typing import Any, Dict, List, Union,Mapping, Optional, TypeVar, Union
from langchain.chains import LLMChain
from langchain.llms.bedrock import Bedrock
from botocore.exceptions import ClientError
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth, helpers
import boto3

logger = logging.getLogger()
logger.setLevel(logging.INFO)

credentials = boto3.Session().get_credentials()

class APIException(Exception):
    def __init__(self, message, code: str = None):
        if code:
            super().__init__("[{}] {}".format(code, message))
        else:
            super().__init__(message)


def handle_error(func):
    """Decorator for exception handling"""

    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except APIException as e:
            logger.exception(e)
            raise e
        except Exception as e:
            logger.exception(e)
            raise RuntimeError(
                "Unknown exception, please check Lambda log for more details"
            )

    return wrapper


### core funcs###########
def aos_knn_search(client, field,q_embedding, index, size=1):
    if not isinstance(client, OpenSearch):   
        client = OpenSearch(
            hosts=[{'host': aos_endpoint, 'port': 443}],
            http_auth = pwdauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )
    query = {
        "size": size,
        "query": {
            "knn": {
                field: {
                    "vector": q_embedding,
                    "k": size
                }
            }
        }
    }
    opensearch_knn_respose = []
    query_response = client.search(
        body=query,
        index=index
    )
    opensearch_knn_respose = [{'idx':item['_source'].get('idx',1),'database_name':item['_source']['database_name'],'table_name':item['_source']['table_name'],'query_desc_text':item['_source']['query_desc_text'],"score":item["_score"]}  for item in query_response["hits"]["hits"]]
    return opensearch_knn_respose

def aos_knn_search_v2(client, field,q_embedding, index, size=1):
    if not isinstance(client, OpenSearch):   
        client = OpenSearch(
            hosts=[{'host': aos_endpoint, 'port': 443}],
            http_auth = pwdauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )
    query = {
        "size": size,
        "query": {
            "knn": {
                field: {
                    "vector": q_embedding,
                    "k": size
                }
            }
        }
    }
    opensearch_knn_respose = []
    query_response = client.search(
        body=query,
        index=index
    )
    opensearch_knn_respose = [{'idx':item['_source'].get('idx',1),'database_name':item['_source']['database_name'],'table_name':item['_source']['table_name'],'exactly_query_text':item['_source']['exactly_query_text'],"score":item["_score"]}  for item in query_response["hits"]["hits"]]
    return opensearch_knn_respose


def aos_reverse_search(client, index_name, field, query_term, exactly_match=False, size=1):
    """
    search opensearch with query.
    :param host: AOS endpoint
    :param index_name: Target Index Name
    :param field: search field
    :param query_term: query term
    :return: aos response json
    """
    if not isinstance(client, OpenSearch):   
        client = OpenSearch(
            hosts=[{'host': aos_endpoint, 'port': 443}],
            http_auth = pwdauth,
            use_ssl=True,
            verify_certs=True,
            connection_class=RequestsHttpConnection
        )
    query = None
    if exactly_match:
        query =  {
            "query" : {
                "match_phrase":{
                    field: {
                        "query": query_term,
                        "analyzer": "ik_smart"
                      }
                }
            }
        }
    else:
        query = {
            "size": size,
            "query": {
                "query_string": {
                "default_field": "exactly_query_text",  
                "query": query_term         
              }
            },
           "sort": [{
               "_score": {
                   "order": "desc"
               }
           }]
    }        
    query_response = client.search(
        body=query,
        index=index_name
    )
    result_arr = [{'idx':item['_source'].get('idx',1),'database_name':item['_source']['database_name'],'table_name':item['_source']['table_name'],'exactly_query_text':item['_source']['exactly_query_text'],"score":item["_score"]}  for item in query_response["hits"]["hits"]]
    return result_arr




def get_vector_by_sm_endpoint(questions, sm_client, endpoint_name):
    parameters = {
    }

    response_model = sm_client.invoke_endpoint(
        EndpointName=endpoint_name,
        Body=json.dumps(
            {
                "inputs": questions,
                "parameters": parameters,
                "is_query" : True,
                "instruction" :  "为这个句子生成表示以用于检索相关文章："
            }
        ),
        ContentType="application/json",
    )
    json_str = response_model['Body'].read().decode('utf8')
    json_obj = json.loads(json_str)
    embeddings = json_obj['sentence_embeddings']
    return embeddings


def k_nn_ingestion_by_aos(docs,index,hostname,username,passwd):
    auth = (username, passwd)
    search = OpenSearch(
        hosts = [{'host': aos_endpoint, 'port': 443}],
        ##http_auth = awsauth ,
        http_auth = auth ,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection
    )
    for doc in docs:
        query_desc_embedding = doc['query_desc_embedding']
        database_name = doc['database_name']
        table_name = doc['table_name']
        query_desc_text = doc["query_desc_text"]
        document = { "query_desc_embedding": query_desc_embedding, 'database_name':database_name, "table_name": table_name,"query_desc_text":query_desc_text}
        search.index(index=index, body=document)
        
def k_nn_ingestion_by_aos_v2(docs,index,hostname,username,passwd):
    auth = (username, passwd)
    search = OpenSearch(
        hosts = [{'host': aos_endpoint, 'port': 443}],
        ##http_auth = awsauth ,
        http_auth = auth ,
        use_ssl = True,
        verify_certs = True,
        connection_class = RequestsHttpConnection
    )
    for doc in docs:
        exactly_query_embedding = doc['exactly_query_embedding']
        database_name = doc['database_name']
        table_name = doc['table_name']
        exactly_query_text = doc["exactly_query_text"]
        document = { "exactly_query_embedding": exactly_query_embedding, 'database_name':database_name, "table_name": table_name,"exactly_query_text":exactly_query_text}
        search.index(index=index, body=document)

    
@handle_error
def lambda_handler(event, context):
    
    embedding_endpoint = os.environ.get('embedding_endpoint')
    region = os.environ.get('region')
    aos_endpoint = os.environ.get('aos_endpoint')
    index_name = os.environ.get('index_name')
    query = event.get('query')
    aos_user = event.get('aos_user')
    aos_pwd = event.get('aos_pwd')
    llm_model_endpoint = os.environ.get('llm_model_endpoint')
    llm_model_name = event.get('llm_model_name', None)
    
    logger.info("embedding_endpoint: {}".format(embedding_endpoint))
    logger.info("region:{}".format(region))
    logger.info("aos_endpoint:{}".format(aos_endpoint))
    logger.info("index_name:{}".format(index_name))
    logger.info("fewshot_cnt:{}".format(fewshot_cnt))
    logger.info("llm_model_endpoint:{}".format(llm_model_endpoint))

    content_handler = ContentHandler()

    embeddings = SagemakerEndpointEmbeddings(
        endpoint_name=embedding_endpoint,
        region_name=region,
        content_handler=content_handler
    )

    logger.info("embedding initialed!")
    auth = AWSV4SignerAuth(credentials, region)
    #auth = (aos_user, aos_pwd)
        
    table_name = None
    aos_client = OpenSearch(
                hosts=[{'host': aos_endpoint, 'port': 443}],
                http_auth = pwdauth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection
            )
    #### reverse 倒排召回 ############
    opensearch_query_response = aos_reverse_search(aos_client, aos_index, "exactly_query_text", query)
    try:
        table_name=opensearch_query_response[0]["table_name"].strip()
    except Exception as e:
        print(e)
        table_name=None
    
    #### reverse 向量召回 ############
    if table_name is None:
        query_embedding = get_vector_by_sm_endpoint(query, sm_client, embedding_endpoint_name)
        opensearch_query_response = aos_knn_search_v2(aos_client, "exactly_query_embedding",query_embedding[0], aos_index, size=10)
        try:
            table_name = responses[0]["table_name"].strip()
        except Exception as e:
            print(e)
            table_name = None
    
    #####使用召回table name执行SqlDataBaseChain#######
    db = SQLDatabase.from_uri(
        "mysql+pymysql://admin:admin12345678@database-us-west-2-demo.cluster-c1qvx9wzmmcz.us-west-2.rds.amazonaws.com/llm",
        include_tables=[table_name], # we include only one table to save tokens in the prompt :)
        sample_rows_in_table_info=0)
    
    db_chain = CustomerizedSQLDatabaseChain.from_llm(llm=bedrock_llm, db=db, verbose=False, return_sql=True)
    response = db_chain.run(query)
        
    return response
