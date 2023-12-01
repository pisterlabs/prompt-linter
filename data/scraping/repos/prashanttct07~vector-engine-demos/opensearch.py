from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3
import os
from sentence_transformers import SentenceTransformer
import sys
from langchain.llms.bedrock import Bedrock
import traceback, socket, datetime

module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock

# Static Section
# Bedrock constants
# os.environ['BEDROCK_ASSUME_ROLE'] = 'arn:aws:iam::706553727873:role/service-role/AmazonSageMaker-ExecutionRole-20211019T121285'
os.environ['AWS_PROFILE'] = 'bedrock_prashant'
os.environ['AWS_DEFAULT_REGION'] = 'us-east-1'

# Load the SentenceTransformer model
model_name = 'sentence-transformers/msmarco-distilbert-base-tas-b'
model = SentenceTransformer(model_name)

# Set the desired vector size
vector_size = 768

# OpenSearch
AWS_PROFILE = "273117053997-us-east-2"
host = '0n2qav61946ja1c7k2a1.us-east-2.aoss.amazonaws.com' # OpenSearch Serverless collection endpoint
region = 'us-east-2' # e.g. us-west-2

host_metrics = 'q31x4gto5pm94bsxiu1g.us-east-2.aoss.amazonaws.com'
# Specify index name for performance logging
perf_logging_index_name = 'aoss-performance-search'

action = {"index": {"_index": perf_logging_index_name}}
actions = []  # prepare bulk request for performance stats
i = 0  # Tracking bulk size for query performance stats


# Bedrock Clients connection
boto3_bedrock = bedrock.get_bedrock_client(os.environ.get('BEDROCK_ASSUME_ROLE', None))

# - create the LLM Model
claude_llm = Bedrock(model_id="anthropic.claude-instant-v1", client=boto3_bedrock, model_kwargs={'max_tokens_to_sample':1000})
titan_llm = Bedrock(model_id= "amazon.titan-tg1-large", client=boto3_bedrock)

# Use this if you need to generate embedding using Titan Embeddings Model.
# from langchain.embeddings import BedrockEmbeddings
# bedrock_embeddings = BedrockEmbeddings(client=boto3_bedrock)
# embedding = np.array(bedrock_embeddings.embed_query(document.page_content))

# - Create Prompts
def get_claude_prompt(context, user_question, knowledgebase_filter):
    if knowledgebase_filter:
        prompt = f"""Human: Answer the question based on the information provided. If the answer is not in the context, say "I don't know, answer not found in the documents."
        <context>
        {context}
        </context>
        <question>
        {user_question}
        </question>
        Assistant:"""
        return prompt
    else:
        prompt = f"""Human: Answer the question as below:"
        <question>
        {user_question}
        </question>
        Assistant:"""
        return prompt

def get_titan_prompt(context, user_question, knowledgebase_filter):
    if knowledgebase_filter:
        prompt = f"""Answer the below question based on the context provided. If the answer is not in the context, say "I don't know, answer not found in the documents".
        {context}
        {user_question}
        """
        return prompt
    else:
        prompt = f"""Answer the question as below:".
        {user_question}
        """
        return prompt



service = 'aoss'
credentials = boto3.Session(profile_name=AWS_PROFILE).get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service,
session_token=credentials.token)

# Create an OpenSearch client
client = OpenSearch(
    hosts = [{'host': host, 'port': 443}],
    http_auth = awsauth,
    timeout = 300,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection
)

# Create an OpenSearch client
client_metrics = OpenSearch(
    hosts = [{'host': host_metrics, 'port': 443}],
    http_auth = awsauth,
    timeout = 300,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection
)


def log_metrics(query, dataset, query_type, took):
    perf_logging_index_name = 'aoss-performance-search'

    action = {"index": {"_index": perf_logging_index_name}}
    actions = []  # prepare bulk request for performance stats
    try:
        # Start Logging
        # Prepare a document to index performance stats
        document = {
            "error": False,
            "@timestamp": datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
            "collection": host,
            "took": took,
            "dataset" : dataset,
            "query_type": query_type,
            "query": query
        }

        # Index Documents
        actions.append(action)
        actions.append(document)
    except:
        print("An exception occurred while processing the request")
        tb = traceback.format_exc()
        print(tb)
        # Prepare a document to index performance stats for failure
        document = {
            "error": True,
            "exception": tb,
            "@timestamp": datetime.datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
            "collection": host,
            "query_type": query_type,
            "query": query
        }

        # Append Documents for error
        actions.append(action)
        actions.append(document)
    client_metrics.bulk(body=actions)




# Define queries for OpenSearch
def query_qna(query, index):
    query_embedding = model.encode(query).tolist()
    query_qna = {
        "size": 3,
        "fields": ["content", "title"],
        "_source": False,
        "query": {
            "knn": {
            "v_content": {
                "vector": query_embedding,
                "k": vector_size
            }
            }
        }
    }

    relevant_documents = client.search(
        body = query_qna,
        index = index
    )
    log_metrics(query, "qna", "semantic", relevant_documents['took'])
    return relevant_documents


# Define queries for OpenSearch
def query_gartner(query, index):
    query_embedding = model.encode(query).tolist()
    query_qna = {
        "size": 3,
        "fields": ["content"],
        "_source": False,
        "query": {
            "knn": {
            "v_content": {
                "vector": query_embedding,
                "k": vector_size
            }
            }
        }
    }

    relevant_documents = client.search(
        body = query_qna,
        index = index
    )
    log_metrics(query, "gartner", "semantic", relevant_documents['took'])
    return relevant_documents



def query_movies(query, sort, genres, rating, index):

    if sort == 'year':
        sort_type = "year"
    elif sort == 'rating':
        sort_type = "rating"
    else:
        sort_type = "_score"

    if genres == '':
        genres = '*'

    if rating == '':
        rating = 0

    query_embedding = model.encode(query).tolist()
    query_knn = {
        "size": 3,
        "sort": [
            {
                sort_type: {
                    "order": "desc"
                }
            }
        ],
        "_source": {
            "includes": [
                "title",
                "plot",
                "rating",
                "year",
                "image_url",
                "genres"
            ]
        },
        "query": {
            "bool": {
                "should": [
                    {
                        "knn": {
                            "v_plot": {
                                "vector": query_embedding,
                                "k": vector_size
                            }
                        }
                    },
                    {
                        "knn": {
                            "v_title": {
                                "vector": query_embedding,
                                "k": vector_size
                            }
                        }
                    }
                ],
                "filter": [
                    {
                        "query_string": {
                            "query": genres,
                            "fields": [
                                "genres"
                            ]
                        }
                    },
                    {
                      "range": {
                        "rating": {
                          "gte": rating
                        }
                      }
                    }
                ]
            }
        }
    }
    response_knn = client.search(
        body = query_knn,
        index = index
    )

    # print (query_knn)
    # print(response_knn)

    # Extract relevant information from the search result
    hits_knn = response_knn['hits']['hits']
    doc_count_knn = response_knn['hits']['total']['value']
    results_knn = [{'genres':  hit['_source']['genres'],'image_url':  hit['_source']['image_url'],'title': hit['_source']['title'], 'rating': hit['_source']['rating'], 'year': hit['_source']['year'], 'plot' : hit['_source']['plot']} for hit in hits_knn]



    query_kw = {
        "size": 3,
        "sort": [
            {
                sort_type: {
                    "order": "desc"
                }
            }
        ],
        "_source": {
            "includes": [
                "title",
                "plot",
                "rating",
                "year",
                "image_url",
                "genres"
            ]
        },
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["plot", "title"]
                    }
                },
                "filter": [
                    {
                        "query_string": {
                            "query": genres,
                            "fields": [
                                "genres"
                            ]
                        }
                    },
                    {
                      "range": {
                        "rating": {
                          "gte": rating
                        }
                      }
                    }
                ]
            }
        }
    }

    response_kw = client.search(
        body = query_kw,
        index = index
    )

    # Extract relevant information from the search result
    hits_kw = response_kw['hits']['hits']
    doc_count_kw = response_kw['hits']['total']['value']
    results_kw = [{'genres':  hit['_source']['genres'],'image_url':  hit['_source']['image_url'],'title': hit['_source']['title'], 'rating': hit['_source']['rating'], 'year': hit['_source']['year'], 'plot' : hit['_source']['plot']} for hit in hits_kw]


    log_metrics(query, "MOVIES", "semantic", response_knn['took'])
    log_metrics(query, "MOVIES", "lexical", response_kw['took'])

    # print (f"Search Results: {search_results}")
    return results_knn, doc_count_knn, results_kw, doc_count_kw

