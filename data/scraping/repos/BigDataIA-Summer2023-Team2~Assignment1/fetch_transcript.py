from airflow.models import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from airflow.models.param import Param
import datetime
from datetime import timedelta
import os
import requests
from airflow.models.baseoperator import chain
from sentence_transformers import SentenceTransformer
import numpy as np
import openai
import redis
from redis.commands.search.field import VectorField, TextField, NumericField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType



# dag declaration
user_input = {
    "company_name": Param(default="LMAT", type='string', minLength=5, maxLength=255),
    "year": Param(default=2015, type='number'),
    "quarter": Param(default=1, type='number'),
    "word_limit": Param(default=500, type='number'),
    "openai_api_key": Param(type='string'),
}


dag = DAG(
    dag_id="fetch_transcript",
    # Run daily midnight to fetch metadata from github
    schedule="0 0 * * *",   # https://crontab.guru/
    start_date=days_ago(0),
    catchup=False,
    dagrun_timeout=timedelta(minutes=60),
    tags=["assignment1", "damg7245", "fetch_transcript"],
)

def folder_names_with_date(year, quarter, company):
    start_month = (quarter - 1) * 3 + 1
    end_month = start_month + 2

    # Create starting and ending datetime objects
    start_date = datetime.datetime(year, start_month, 1)
    end_date = datetime.datetime(year, end_month, 1) + datetime.timedelta(days=31)

    # Generate the range of dates
    folder_names = []
    current_date = start_date
    while current_date <= end_date:
        folder_names.append(current_date.strftime("%Y-%m-%d").replace('-','')+'_'+company)
        current_date += datetime.timedelta(days=1)
    return folder_names

def get_words_github(ti, **kwargs):
    company_name = kwargs['params']['company_name']
    year = kwargs['params']['year']
    quarter = kwargs['params']['quarter']
    word_limit = kwargs['params']['word_limit']

    folder_names = folder_names_with_date(year, quarter, company_name)
    ti.xcom_push(key="folder_names", value=folder_names)

    words = []

    for folder in folder_names:
      url = f"{os.getenv('DATASET_GITHUB_URL')}/MAEC_Dataset/{folder}/text.txt"
      page = requests.get(url)
      if page.status_code != 200:
        continue
      words += page.text.split()

    first_n_words = ' '.join(words[:word_limit])
    return first_n_words

def generate_sbert_embeddings(ti):
    model = SentenceTransformer(os.getenv('SBERT_MODEL','sentence-transformers/all-MiniLM-L6-v2'))
    words_to_encode = ti.xcom_pull(key="return_value", task_ids='get_words_github')
    print(words_to_encode)
    ti.xcom_push(key="words_to_encode", value=words_to_encode)
    embeddings = model.encode(words_to_encode)
    print(type(embeddings), embeddings)
    # vector = np.array(embeddings).astype(np.float32).tobytes()
    # print(type(vector), vector)
    return embeddings.tolist()
    
def generate_openai_embeddings(ti, **kwargs):
    openai.api_key = kwargs["params"]["openai_api_key"]
    model_id = os.getenv("OPENAI_ENGINE", "text-embedding-ada-002")
    words_to_encode = ti.xcom_pull(key="return_value", task_ids='get_words_github')
    
    embeddings = openai.Embedding.create(
        input=words_to_encode,
        engine=model_id)['data'][0]['embedding']
    return embeddings

def save_data_to_redis(ti, **kwargs): 
    company_name = kwargs['params']['company_name'] 
    year = kwargs['params']['year'] 
    quarter = kwargs['params']['quarter'] 
     
    plain_text = ti.xcom_pull(key="return_value", task_ids='get_words_github') 
    sbert_embeddings = ti.xcom_pull(key="return_value", task_ids='generate_sbert_embeddings') 
    sbert_vector = np.array(sbert_embeddings).astype(np.float32).tobytes()

    openai_embeddings = ti.xcom_pull(key="return_value", task_ids='generate_openai_embeddings') 
    openai_vector = np.array(openai_embeddings).astype(np.float32).tobytes()
    r = redis.Redis(host=os.getenv("REDIS_DB_HOST", 'redis-stack'),  # Local redis error 
                    port= os.getenv("REDIS_DB_PORT", "6379"), 
                    username=os.getenv("REDIS_DB_USERNAME", ""), 
                    password=os.getenv("REDIS_DB_PASSWORD", ""), 
                    decode_responses=True 
                    ) 
    folder_names = ti.xcom_pull(key="folder_names", task_ids='get_words_github') 
    datekey = folder_names[0]

    year= datekey.split('_')[0][:4]
    month= datekey.split('_')[0][4:6]
    date= datekey.split('_')[0][-2:]

    data = { 
        "date" : date,
        "month": month,
        "year" : year, 
        "quarter" : quarter, 
        "company_ticker": company_name,
        "plain_text" : plain_text, 
        "sbert_embeddings": sbert_vector, 
        "openai_embeddings": openai_vector 
    } 
    SCHEMA = [
        NumericField("date"),
        NumericField("month"),
        NumericField("year"),
        TextField("company_ticker"),
        TextField("plain_text"),
        VectorField("sbert_embeddings", "FLAT", {"TYPE": "FLOAT32", "DIM": 384, "DISTANCE_METRIC": "COSINE"}),
        VectorField("openai_embeddings", "FLAT", {"TYPE": "FLOAT32", "DIM": 1536, "DISTANCE_METRIC": "COSINE"}),
         ]
    r.hset(f"post:{company_name}:{year}_{quarter}", mapping=data) 
    
    if r.exists("embeddings"):
        r.ft.drop_index("embeddings")
        # r.zadd("embeddings", SCHEMA)
        
    try:
        r.ft("embeddings").create_index(fields=SCHEMA, definition=IndexDefinition(prefix=["post:"], index_type=IndexType.HASH))
    except Exception as e:
        print("Index already exists")

    r.close()
   
    # Create the index
    return "Data saved to redis" 

with dag:
    get_data_from_github_task = PythonOperator(
        task_id='get_words_github',
        python_callable=get_words_github,
        provide_context=True,
        dag=dag,
    )
    
    generate_sbert_embeddings_task = PythonOperator(
        task_id='generate_sbert_embeddings',
        python_callable=generate_sbert_embeddings,
        provide_context=True,
        dag=dag,
    )

    generate_openai_embeddings_task = PythonOperator(
        task_id='generate_openai_embeddings',
        python_callable=generate_openai_embeddings,
        provide_context=True,
        dag=dag,
    )

    save_data_to_redis_task = PythonOperator( 
        task_id='save_data_to_redis', 
        python_callable=save_data_to_redis, 
        provide_context=True, 
        dag=dag, 
    ) 
    chain(get_data_from_github_task, [generate_openai_embeddings_task, generate_sbert_embeddings_task], save_data_to_redis_task) 

# {
#   "company_name": "LMAT",
#   "year": 2015,
#   "quarter": 1,
#   "word_limit": 500
# }