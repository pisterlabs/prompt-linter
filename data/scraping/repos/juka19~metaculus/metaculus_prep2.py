import logging
import sqlite3
import urllib.request
import json
import re
import os
from datetime import datetime, timedelta
import time
from langchain.llms import self_hosted 

from langchain.prompts import PromptTemplate
from langchain import HuggingFaceHub
from langchain.chains import SequentialChain, LLMChain
from langchain.callbacks import FileCallbackHandler
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.log.logging_mixin import LoggingMixin
from langchain.schema import prompt_template, output_parser
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from huggingface_hub import login
from numpy import random, rec
from numpy.core import numeric
import pandas as pd
from cerberus import Validator
from fp.fp import FreeProxy
from fp.errors import FreeProxyException
from sqlalchemy.engine import result
from sqlalchemy.orm.descriptor_props import DescriptorProperty
from metaculus_prep import validate_input, write_json, write_json
from pydantic import BaseModel, Field, validator

with open("../../assets/HF_API_TOKEN.txt", "r") as f:
    hf_token = f.read()

def read_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

date_answer = ResponseSchema(name='date_answer', description='The answer to the instructions, parsed as a date in the format YY-MM-DD')
binary_answer = ResponseSchema(name='binary_answer', description='The answer to the instructions, parsed as a binary answer (Yes/No)')
numerical_answer = ResponseSchema(name='numerical_answer', description='The answer to the instructions, parsed as a numerical answer')

def langchain_pipeline_no_expert(record, llm=None):
    result = {}
    LoggingMixin().log.info(f'Running langchain pipeline for record {record["id"]}')
    if llm is None:
        llm = HuggingFaceHub(
            repo_id="tiiuae/falcon-7b-instruct",
            huggingfacehub_api_token=hf_token,
            model_kwargs={"pad_token_id": 11,
            "max_length": 10000,
            "do_sample": True,
            "top_k": 10,
            "num_return_sequences": 1,
            "trust_remote_code": True}
        )
    if record['forecast_type'] == 'binary':
        
        parser = StructuredOutputParser.from_response_schemas([binary_answer])
        
        template = """
        Please answer the following question. Take all information into consideration and answers this question either with Yes or with No, nothing else.
        {context}
        {question}
        [Agent Answer]:
        """.strip()
        
        chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                template=template, 
                input_variables=['question', 'context'], 
                # partial_variables={"format_instructions": parser.get_format_instructions()},
                # parser=parser
            ),
        )
        
        if record['description'] == '':
            context = record['res_criteria']
        else:
            context = record['description']
        
        args = {"question": record['title'], "context": context}
    
    elif record['forecast_type'] == 'date_range':
        
        parser = StructuredOutputParser.from_response_schemas([date_answer])
        
        template = """
        Please answer the following question. Take all information into consideration and answer this question only with an exact date estimate in the format of YY-MM-DD and no other characters.
        The maximum date is {max} and the minimum date is {min}.
        {context}
        {question}
        [Agent Answer]:
        """.strip()
        
        chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                template=template, 
                input_variables=['question', 'context', 'max', 'min'], 
                # partial_variables={"format_instructions": parser.get_format_instructions()},
                # parser=parser
            ),
        )
        
        if record['description'] == '':
            context = record['res_criteria']
        else:
            context = record['description']
        
        args = {"question": record['title'], "context": context, "max": record["possibilities"]["scale"]['max'], "min": record["possibilities"]["scale"]['min']}
        
        
    elif record['forecast_type'] == 'numerical':
        
        parser = StructuredOutputParser.from_response_schemas([numerical_answer])
        
        template = """
        Please answer the following question. Take all information into consideration and answer this question with a numerical estimate. Only use numbers and no other characters.
        The maximum number is {max} and the minimum number is {min}.
        {context}
        {question}
        [Agent Answer]:
        """.strip()
        
        chain = LLMChain(
            llm=llm,
            prompt=PromptTemplate(
                template=template, 
                input_variables=['question', 'context', 'max', 'min'], 
                # partial_variables={"format_instructions": parser.get_format_instructions()},
                # parser=parser
            ),
        )
        
        if record['description'] == '':
            context = record['res_criteria']
        else:
            context = record['description']
        
        args = {"question": record['title'], "context": context, "max": record["possibilities"]["scale"]['max'], "min": record["possibilities"]["scale"]['min']}
        
    pred = chain.predict(**args)
    
    
    result['answer'] = pred
    result['id'] = record['id']
    result = {**result, **args}
    return result

def run_langchain_pipeline(data, llm):
    results = []
    for record in data:
        LoggingMixin().log.info(f'Running langchain pipeline for record {record["id"]}')
        result = langchain_pipeline_no_expert(record, llm=llm)
        results.append(result)
    return results

v2 = Validator(
    {
        'answer': {'type': ['string', 'number']},
        'context': {'type': 'string'},
        'question': {'type': 'string'},
        'id': {'type': 'integer'}
    },
    allow_unknown=True
)

def validate_output(data):
    for output in data:
        LoggingMixin().log.info(f'Validating output for record {output["id"]}')
        if not v2.validate(output):
            raise ValueError(v2.errors)
    return data

llm = HuggingFaceHub(
    repo_id="meta-llama/Llama-2-7b-chat-hf",
    model_kwargs={'max_num_batch_tokens':4096, 'max_tokens': 4096, 'max_token_limit': 4096},
    huggingfacehub_api_token=hf_token
)

dag = DAG(
    dag_id="metaculus_prep2",
    schedule_interval='@once', 
    default_args={
        'owner': 'airflow',
        'retries': 1,
        'retry_delay': timedelta(minutes=5),
        'start_date': datetime.today(),
        },
    catchup=False,
)

with dag:
    task1 = PythonOperator(
        task_id='read_json',
        python_callable=read_json,
        op_kwargs={'json_path': '../../data/all_data.json'}
    )
    task2 = PythonOperator(
        task_id='run_langchain_pipeline',
        python_callable=run_langchain_pipeline,
        op_kwargs={'data': task1.output, 'llm': llm}
    )
    task3 = PythonOperator(
        task_id='validate_output',
        python_callable=validate_output,
        op_kwargs={'data': task2.output}
    )
    task4 = PythonOperator(
        task_id='write_json',
        python_callable=write_json,
        op_kwargs={'data': task2.output, 'output_path': '../../data/preds_no_expert.json'}
    )
    task1 >> task2 >> task3 >> task4
