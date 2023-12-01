from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from google.cloud import storage
from google.oauth2 import service_account
import os
import pymysql
import whisper
import openai
import json
from pydub import AudioSegment
import pendulum
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.providers.mysql.hooks.mysql import MySqlHook
from airflow.hooks.base_hook import BaseHook
from google.auth.credentials import AnonymousCredentials
from airflow.models.param import Param

openai.api_key = os.getenv("open_api_key")
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 3, 28),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    'damg7245-a4-pipeline_batch',
    default_args=default_args,
    description='Transcription of recording',
    schedule_interval=timedelta(minutes=5),
    catchup=False
)

#Util functions
def init_gcp_bucket():
    # Get the credentials from Airflow Admin Connection'
    your_gcp_keys = { 
        "type": os.environ.get('type'),
        "project_id": os.environ.get('project_id'),
        "private_key_id": os.environ.get('private_key_id'),
        "private_key": os.environ.get('private_key').replace('\\n', '\n'),
        "client_email": os.environ.get('client_email'),
        "client_id": os.environ.get('client_id'),
        "auth_uri": os.environ.get('auth_uri'),
        "token_uri": os.environ.get('token_uri'),
        "auth_provider_x509_cert_url": os.environ.get('auth_provider_x509_cert_url'),
        "client_x509_cert_url": os.environ.get('client_x509_cert_url')
    }
    
    # Set the credentials
    credentials = service_account.Credentials.from_service_account_info(your_gcp_keys)
    storage_client = storage.Client(credentials=credentials)
    return storage_client

def upload_objects(folder,**kwargs):
    # ti = kwargs['ti']
    # file_name = ti.xcom_pull(key='file_name', task_ids=['download_recording'])[0]
    files= list_all_file()
    for file in files:
        folder += f'{file}.txt'
        object_n = f'{file}.txt'
        storage_client=init_gcp_bucket()
        bucket = storage_client.get_bucket(os.getenv("bucket_name")) 
        blob = bucket.blob(folder)
        blob.upload_from_filename(object_n)

def list_all_file():
    files_list=[]
    storage_client=init_gcp_bucket()
    # Get the bucket object
    bucket = storage_client.get_bucket(os.getenv("bucket_name")) 
    # Iterate through all the blobs in the bucket
    for blob in bucket.list_blobs(prefix='recording/'):
        # Print the name of the file
       files_list.append(str(blob.name).split('/')[1].split('.')[0])
    return files_list

def get_transcripts_objects(file_name):
    storage_client=init_gcp_bucket()
    bucket = storage_client.get_bucket(os.getenv("bucket_name")) 
    blob_name = f"transcript/{file_name}.txt"
    blob=bucket.blob(blob_name)
    return blob.download_as_string()

def write_database(Recording_Name,Q1,Q2,Q3,Q4):
    try:  
        conn = pymysql.connect(
                host = os.getenv("host"), 
                user = os.getenv("user"),
                password = os.getenv("password"),
                db = os.getenv("db"))
        cursor = conn.cursor()
        sql_insert=f"INSERT INTO Recording_Details ( Recording_Name , Question1 , Question2, Question3,Question4) VALUES (%s, %s ,%s, %s, %s);"
        record=(Recording_Name,Q1,Q2,Q3,Q4)
        cursor.execute(sql_insert,record)
        conn.commit()
        cursor.close()
    except Exception as error:
        print("Failed to insert record into table {}".format(error))
    # finally:
    #     move_recording()

def move_recording(**kwargs):
    ti = kwargs['ti']
    recording_name = ti.xcom_pull(key='file_name', task_ids=['download_recording'])[0]
    storage_client=init_gcp_bucket()
    bucket = storage_client.get_bucket(os.getenv("bucket_name"))
    blob_name = f"recording/{recording_name}.mp3"
    source_blob = bucket.blob(blob_name)
    # copy to new destination
    bucket.copy_blob(source_blob, bucket, f"processed/{recording_name}.mp3")
    # delete in old destination
    source_blob.delete()
        
def get_recordings_objects(**kwargs):
    #recording_name = kwargs['dag_run'].conf['recording_name']
    recordings=list_all_file()
    storage_client=init_gcp_bucket()
    file_names=[]
    for recording_name in recordings:
        print(recording_name)
        bucket = storage_client.get_bucket(os.getenv("bucket_name"))
        blob_name = f"recording/{recording_name}.mp3"
        blob=bucket.blob(blob_name)
        blob.download_to_filename(f"{recording_name}.mp3")
    ti = kwargs['ti']
    ti.xcom_push(key='file_name', value=recordings)

def transcribe_audio(**kwargs):
    # Convert the MP3 file to WAV format
    ti = kwargs['ti']
    #file_path = ti.xcom_pull(key='file_name', task_ids=['download_recording'])[0]
    recordings=list_all_file()
    transcribe_list=[]
    os.environ["PATH"] += os.pathsep + '/usr/bin/ffmpeg'
    for recording in recordings:
        sound = AudioSegment.from_mp3(f"{recording}.mp3")
        sound.export(f'/tmp/{recording}.wav', format= 'wav')
        model_id = 'whisper-1'
        with open(f'/tmp/{recording}.wav','rb') as audio_file:
            transcription=openai.Audio.transcribe(api_key=openai.api_key, model=model_id, file=audio_file, response_format='text')
            file_text = open(f"{recording}.txt", "w")
            file_text.write(transcription)
            transcribe_list.append(transcription)
    ti.xcom_push(key='transcripts', value=transcribe_list)

def chat_gpt(query,prompt):
    response_summary =  openai.ChatCompletion.create(
        model = "gpt-3.5-turbo", 
        messages = [
            {"role" : "user", "content" : f'{query} {prompt}'}
        ]
    )
    return response_summary['choices'][0]['message']['content']
    
def query_chat_gpt(**kwargs):
    #global transcript 
    ti = kwargs['ti']
    prompts = ti.xcom_pull(key='transcripts', task_ids=['transcribe_audio'])[0]
    print(prompts)
    recordings=list_all_file()
    for i,prompt in enumerate(prompts):
        #Query1
        query1='give the summary in 700 character: '
        q1=chat_gpt(prompt,query1)
        ##Query2
        query2="what is the mood or emotion in the text in less than 700 character? "
        q2=chat_gpt(prompt,query2)
        ##Query3
        query3="what are the main keywords in less than 700 character? "
        q3=chat_gpt(prompt,query3)
        ##Query4
        query4="What should be the next steps in less than 700 character?"
        q4=chat_gpt(prompt,query4)
        file_name=ti.xcom_pull(key='file_name', task_ids=['download_recording'])[0]
        write_database(recordings[i],q1,q2,q3,q4)
        os.remove(f"{recordings[i]}.mp3")
        os.remove(f"{recordings[i]}.txt")
        #remove file from object store
        storage_client=init_gcp_bucket()
        bucket = storage_client.get_bucket(os.getenv("bucket_name"))
        blob_name = f"recording/{recordings[i]}.mp3"
        source_blob = bucket.blob(blob_name)
        # copy to new destination
        bucket.copy_blob(source_blob, bucket, f"processed/{recordings[i]}.mp3")
        # delete in old destination
        source_blob.delete()
  
def upsert_text_embedings(**kwargs):
    import pandas as pd
    import pinecone
    index_name = os.environ.get('pinecone_index')
    # initialize connection to pinecone (get API key at app.pinecone.io)
    pinecone.init(
        api_key=os.environ.get('pinecone_api_key'),
        environment=os.environ.get('pinecone_env')  # find next to api key in console
    )
    # check if 'openai' index already exists (only create index if not)
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=1536)
    # connect to index
    index = pinecone.Index(index_name)
    MODEL = os.environ.get('openai_embedings')
    files = get_recordings_objects()
    print(files)
    for file in files:
        batch_size = 32 
        recording_id=file.split('-id-')[1]
        for i in range(0, len(files), batch_size):
            # set end position of batch
            i_end = min(i+batch_size, len(files))
            # get batch of lines and IDs
            lines_batch = files[i: i+batch_size]
            ids_batch = [str(n) for n in range(i, i_end)]
            # create embeddings
            res = openai.Embedding.create(input=lines_batch, engine=MODEL)
            embeds = [record['embedding'] for record in res['data']]
            # prep metadata and upsert batch
            meta = [recording_id]
            #meta = df['business_id'][i: i_end].reset_index(drop=True).apply(lambda x: {'business_id': x}).to_list()
            to_upsert = zip(ids_batch, embeds, meta)
            # upsert to Pinecone
            index.upsert(vectors=list(to_upsert),namespace=recording_id)
        write_database(recording_id)

  
t0 = BashOperator(
    task_id='install_ffmpeg',
    bash_command='sudo apt-get update && sudo apt-get -y install ffmpeg',
    dag=dag)

t1 = PythonOperator(
    task_id='download_recording',
    python_callable=get_recordings_objects,
    dag=dag,
    provide_context=True,
)

t2 = PythonOperator(
    task_id='transcribe_audio',
    python_callable=transcribe_audio,
    dag=dag,
    provide_context=True,
)

t3 = PythonOperator(
    task_id='upload_transcript',
    python_callable=upload_objects,
    op_kwargs={'folder': 'transcript/'},
    dag=dag,
    provide_context=True,
)

t4 = PythonOperator(
    task_id='upsert_text_embedings',
    python_callable=upsert_text_embedings,
    dag=dag,
    provide_context=True,
)

t5 = PythonOperator(
    task_id='query_chat_gpt',
    python_callable=query_chat_gpt,
    dag=dag,
    provide_context=True,
)

t0 >> t1 >> t2 >> t3 >> t4 >> t5