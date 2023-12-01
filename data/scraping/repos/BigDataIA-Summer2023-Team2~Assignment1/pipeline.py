from utils import schemas
import requests
import os
from requests.auth import HTTPBasicAuth
import uuid
import redis
import openai
from sentence_transformers import SentenceTransformer
from redis.commands.search.query import Query
import numpy as np
import logging
import json

AIRFLOW_API_URL = os.getenv("AIRFLOW_API_URL", "http://localhost:8080/api/v1")

logger = logging.getLogger(__name__)

def trigger_fetch_transcript(userInput: schemas.EmbeddTranscript):
    url = f"{AIRFLOW_API_URL}/dags/fetch_transcript/dagRuns"
    data = {
        "conf": {
                "company_name": userInput.company_name,
                "year": userInput.year,
                "quarter": userInput.quarter,
                "word_limit": userInput.word_limit,
                "openai_api_key": userInput.openai_api_key
            },
        "dag_run_id": f"fetch_transcript_{uuid.uuid4().hex}",
        }
    response = requests.request("POST", url, auth=HTTPBasicAuth('airflow', 'airflow'), json=data)
    if response.status_code != 200:
        return {"message": f"Internal Server Error {response.text}"}
    return {"message": "Fetching transcript", "details": response.text}

def fetch_transcript(userInput: schemas.FetchTranscript):
    redis_client = redis.Redis(host=os.getenv("REDIS_DB_HOST", 'redis-stack'),  # Local redis error 
                port= os.getenv("REDIS_DB_PORT", "6379"), 
                username=os.getenv("REDIS_DB_USERNAME", ""), 
                password=os.getenv("REDIS_DB_PASSWORD", ""), 
                )
    # Fetch a single key
    key= 'post:'+str(userInput.company_name)+':'+str(userInput.year)+'_'+str(userInput.quarter)
    data = redis_client.hgetall(key)
    # Convert byte strings to regular strings
    decoded_data = {key.decode('latin-1'): value.decode('latin-1') for key, value in data.items()}
    # decoded_data = {base64.b64decode(key).decode("latin-1"): base64.b64decode(value).decode("latin-1") for key, value in data.items()}
    response = {"Transcript": json.dumps(decoded_data.get('plain_text'))}
    return response

def trigger_fetch_metadata_dag():
    url = f"{AIRFLOW_API_URL}/dags/metadata_load/dagRuns"
    data = {
        "dag_run_id": f"metadata_load_{uuid.uuid4().hex}",
        }
    response = requests.request("POST", url, auth=HTTPBasicAuth('airflow', 'airflow'), json=data)
    if response.status_code != 200:
        return {"message": f"Internal Server Error {response.text}"}
    return {"message": "Fetching transcript", "details": response.text}

def get_vss_results(query_string, embedding_type, openai_api_key):
    # Redis connection details
   
    redis_client = redis.Redis(host=os.getenv("REDIS_DB_HOST", 'redis-stack'),  # Local redis error 
                    port= os.getenv("REDIS_DB_PORT", "6379"), 
                    username=os.getenv("REDIS_DB_USERNAME", ""), 
                    password=os.getenv("REDIS_DB_PASSWORD", ""), 
                    decode_responses=True 
                    )
    if embedding_type=='openai':
        # Vectorize the query using OpenAI's text-embedding-ada-002 model
        openai.api_key = openai_api_key
        print("Vectorizing query...")
        model_id="text-embedding-ada-002"
        openaiembed = openai.Embedding.create(
            input=query_string,
            engine=model_id)
        query_vector = openaiembed["data"][0]["embedding"]
        # Convert the vector to a numpy array
        query_vector = np.array(query_vector).astype(np.float32).tobytes()
        base_query = "*=>[KNN 5 @openai_embeddings $vector AS vector_score]"
    
    if embedding_type=='sbert':
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode(query_string)
        
        # Convert the vector to a numpy array
        query_vector = np.array(embeddings).astype(np.float32).tobytes()
        base_query = "*=>[KNN 5 @sbert_embeddings $vector AS vector_score]"

    query = Query(base_query).return_fields("plain_text", "vector_score").sort_by("vector_score").dialect(2)    
 
    try:
        results = redis_client.ft("embeddings").search(query, query_params={"vector": query_vector})
    except Exception as e:
        print("Error calling Redis search: ", e)
        return None
    return process_result(results)

def get_vss_hybrid_year_results(query_string, start_year, end_year, api, apikey):
     # Redis connection details
    redis_client = redis.Redis(host=os.getenv("REDIS_DB_HOST", 'redis-stack'),  # Local redis error 
                port= os.getenv("REDIS_DB_PORT", "6379"), 
                username=os.getenv("REDIS_DB_USERNAME", ""), 
                password=os.getenv("REDIS_DB_PASSWORD", ""), 
                decode_responses=True 
                )
    if api=='openai':
        # Vectorize the query using OpenAI's text-embedding-ada-002 model
        openai.api_key = apikey
        model_id="text-embedding-ada-002"
        openaiembed = openai.Embedding.create(
            input=query_string,
            engine=model_id)
        query_vector = openaiembed["data"][0]["embedding"]
        # Convert the vector to a numpy array
        query_vector = np.array(query_vector).astype(np.float32).tobytes()
        base_query = "(@year:["+start_year+" "+end_year+"])=>[KNN 5 @openai_embeddings $vector AS vector_score]"

    if api=='sbert':
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode(query_string)
        # Convert the vector to a numpy array
        query_vector = np.array(embeddings).astype(np.float32).tobytes()
        base_query = "(@year:["+str(start_year)+" "+str(end_year)+"])=>[KNN 5 @sbert_embeddings $vector AS vector_score]"


    print(base_query)
    query = Query(base_query).return_fields("plain_text", "vector_score").sort_by("vector_score").dialect(2)    

    try:
        results = redis_client.ft("embeddings").search(query, query_params={"vector": query_vector})
    except Exception as e:
        print("Error calling Redis search: ", e)
        return None
    return process_result(results)

def get_vss_hybrid_company_results(query_string,company,api,apikey):
    # Redis connection details
    redis_client = redis.Redis(host=os.getenv("REDIS_DB_HOST", 'redis-stack'),  # Local redis error 
                    port= os.getenv("REDIS_DB_PORT", "6379"), 
                    username=os.getenv("REDIS_DB_USERNAME", ""), 
                    password=os.getenv("REDIS_DB_PASSWORD", ""), 
                    decode_responses=True 
                    )
    if api=='openai':
        # Vectorize the query using OpenAI's text-embedding-ada-002 model
        openai.api_key = apikey
        model_id="text-embedding-ada-002"
        openaiembed = openai.Embedding.create(
            input=query_string,
            engine=model_id)
        query_vector = openaiembed["data"][0]["embedding"]
        # Convert the vector to a numpy array
        query_vector = np.array(query_vector).astype(np.float32).tobytes()
        base_query = "(@company_ticker:"+company+")=>[KNN 5 @openai_embeddings $vector AS vector_score]"

    if api=='sbert':
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode(query_string)
        # Convert the vector to a numpy array
        query_vector = np.array(embeddings).astype(np.float32).tobytes()
        base_query = "(@company_ticker:"+company+")=>[KNN 5 @sbert_embeddings $vector AS vector_score]"


    print(base_query)
    query = Query(base_query).return_fields("plain_text", "vector_score").sort_by("vector_score").dialect(2)    

    try:
        results = redis_client.ft("embeddings").search(query, query_params={"vector": query_vector})
    except Exception as e:
        print("Error calling Redis search: ", e)
        return None
    return process_result(results)

def process_result(results):
    response = []
    for i, embedd in enumerate(results.docs):
        score = 1 - float(embedd.vector_score)
        print(f"\t{i}. {embedd.plain_text} (Score: {round(score ,3) })")
        logger.info(f"\t{i}. {embedd.plain_text} (Score: {round(score ,3) })")
        response.append(f"\t{i}. {embedd.plain_text} (Score: {round(score ,3) })")
    return response
