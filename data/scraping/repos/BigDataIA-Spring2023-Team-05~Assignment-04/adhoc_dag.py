# %% 
from airflow.models import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from airflow.utils.dates import days_ago
from airflow.models.param import Param
from datetime import timedelta
from pathlib import Path

import sys
sys.path.append('/opt/airflow/common_package/')

from openai_gpt import OpenAIGPT
from aws_s3_bucket import AWSS3Download
from audio_transcribe import AudioTranscribe


aws_cloud = AWSS3Download()
audio_transcribe = AudioTranscribe()
open_ai_gpt = OpenAIGPT()



# %%
dag = DAG(
    dag_id="adhoc",
    schedule= None,   # https://crontab.guru/
    start_date=days_ago(0),
    catchup=False,
    dagrun_timeout=timedelta(minutes=60),
    tags=["damg7245"],
)



with dag:


    get_audio_files_from_s3 = PythonOperator(
        task_id='get_audio_file_from_s3',
        python_callable= aws_cloud.get_all_adhoc_files,
        provide_context=True,
        do_xcom_push=True,
        dag=dag,
    )


    transcribe_audio = PythonOperator(
        task_id='transcribe_audio',
        python_callable= audio_transcribe.transcribe_adhoc_audio_link,
        provide_context=True,
        do_xcom_push=True,
        dag=dag,
    )

    moving_transcription_to_aws_bucket = PythonOperator(
        task_id='moving_transcription_to_aws_bucket',
        python_callable= aws_cloud.move_adhoc_audio_with_transcription,
        op_kwargs={"text": "{{ ti.xcom_pull(task_ids='transcribe_audio')}}"},
        provide_context=True,
        dag=dag,
    )

    moving_audio_file_to_proccessd_aws_bucket = PythonOperator(
        task_id='moving_audio_file_to_proccessd_aws_bucket',
        python_callable= aws_cloud.move_file_to_adhoc_processes_folder,
        provide_context=True,
        dag=dag,
    )


    generate_default_questions_for_transcription = PythonOperator(
        task_id='generate_default_questions_for_transcription',
        python_callable= open_ai_gpt.generate_questions_for_transcribed_text,
        op_kwargs={"text": "{{ ti.xcom_pull(task_ids='transcribe_audio')}}"},
        provide_context=True,
        dag=dag,
    )



    # Flow
    get_audio_files_from_s3 >> transcribe_audio >> [moving_transcription_to_aws_bucket, moving_audio_file_to_proccessd_aws_bucket] >> generate_default_questions_for_transcription
    # get_all_audio_files_from_s3 >> transcribe_audio