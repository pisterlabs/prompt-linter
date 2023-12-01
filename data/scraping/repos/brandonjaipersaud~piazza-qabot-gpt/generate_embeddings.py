"""Embed model and/or instructor answers"""
import pandas as pd
import sys
import time

from csv_manipulation.csv_utils import add_column, split_answers_csv

from utils.utils import * 
from openai_interface import prompt_gpt3
from cohere_interface.cohere_api import cohere_embeddings
from utils.parse_answers import *

import argparse
import openai
import cohere



def add_embedding(row, embedding_type, model='text-embedding-ada-002', embedding_api='cohere', cohere_api_key = None):

    """Add both instructor and model embeddings if the respective answers exist. Independent from Piazza API"""

    assert(embedding_type == 'instructor' or embedding_type == 'model')
    if embedding_type == 'instructor':
        instructor_answer = row['instructor_answer']
        if not is_empty(instructor_answer):
            instructor_answer = instructor_answer.replace("\n", " ")
            if embedding_api == 'openai' or embedding_api == 'both':
                try:
                    instructor_answer_embeddings = prompt_gpt3.gpt3_embeddings(instructor_answer, print_response=False)
                except Exception:
                        print('Sleeping for 20')
                        time.sleep(20)
                        model_answer_embeddings = prompt_gpt3.gpt3_embeddings(instructor_answer, print_response=False)
                row[f'instructor_answer_embeddings_openai'] = instructor_answer_embeddings
            
            if embedding_api == 'cohere' or embedding_api == 'both':
                instructor_answer_embeddings = cohere_embeddings(instructor_answer, cohere_api_key)
                row[f'instructor_answer_embeddings_cohere'] = instructor_answer_embeddings



    elif embedding_type == 'model':    
        model_answer = row['model_answer'] 
        model_answer_embeddings = ""
        if not is_empty(model_answer):
            if embedding_api == 'openai' or embedding_api == 'both':
                model_answer = model_answer.replace("\n", " ")
                try:
                    print(f'Attempting to embed:\n {model_answer} \nusing {embedding_api}')
                    model_answer_embeddings = prompt_gpt3.gpt3_embeddings(model_answer, print_response=False)
                    
                except Exception:
                    print('Sleeping for 20')
                    time.sleep(20)
                    print(f'Reattempting to embed:\n {model_answer} \nusing {embedding_api}')
                    model_answer_embeddings = prompt_gpt3.gpt3_embeddings(model_answer, print_response=False)
                
                row[f'model_answer_embeddings_openai'] = model_answer_embeddings

            if embedding_api == 'cohere' or embedding_api == 'both':
                try:
                    print(f'Attempting to embed:\n {model_answer} \nusing {embedding_api}')
                    model_answer_embeddings = cohere_embeddings(model_answer, cohere_api_key)
                except Exception:
                    print('Sleeping for 45')
                    time.sleep(45)
                    print(f'Reattempting to embed:\n {model_answer} \nusing {embedding_api}')
                    model_answer_embeddings = cohere_embeddings(model_answer, cohere_api_key)

                row[f'model_answer_embeddings_cohere'] = model_answer_embeddings

     
    else:
        print('Invalid embedding type specified')

    return row

def embed_answers(df:pd.DataFrame, args, force=True, num_embeddings=1, embedding_api='cohere'):
    """Create instructor + model answer embeddings"""
    cohere_api_key = None
    if args.api == 'cohere' or args.api == 'both':
        with open(args.cohere_api_key_path, 'r') as f:
            cohere_api_key = f.readline()

    if args.i:
        if embedding_api == 'cohere' or embedding_api == 'both':
            df, should_process_instructor = add_column('instructor_answer_embeddings_cohere', df, force)

        if embedding_api == 'openai' or embedding_api == 'both':
            df, should_process_instructor = add_column('instructor_answer_embeddings_openai', df, force)

        if should_process_instructor:
            df = df.apply(lambda row: add_embedding(row, 'instructor', embedding_api=args.api, cohere_api_key=cohere_api_key), axis=1)
            

    if args.m:
        if embedding_api == 'cohere' or embedding_api == 'both':
            df, should_process_model = add_column('model_answer_embeddings_cohere', df, force)

        if embedding_api == 'openai' or embedding_api == 'both':
            df, should_process_model = add_column('model_answer_embeddings_openai', df, force)
           
        if should_process_model:
            df = df.apply(lambda row: add_embedding(row, 'model', embedding_api=args.api, cohere_api_key=cohere_api_key), axis=1)

    return df



def check_columns(df, args):
    if args.i and "instructor_answer" not in df.columns:
        print('Instructor_answer embeddings set and instructor answer not in df!')
        return False 

    if args.m and "model_answer" not in df.columns:
        print('Model_answer embeddings set and model answer not in df!')
        return False 
    
    return True



def parse_args(args, df):
    if args.api == 'openai' or args.api == 'both':
        openai.api_key_path = args.openai_api_key_path

    # cohere api key loading done within above function

    # cohere_api_key = None
    # if args.api == 'cohere' or args.api == 'both':
    #     with open(args.cohere_api_key_path, 'r') as f:
    #         cohere_api_key = f.readline()
 

    if args.size >= 0:
        df = df.head(args.size) 
    
    if args.split:
        df = split_answers_csv(df)

    if not args.api:
        args.api = 'both'


    if check_columns(df, args):
        df = embed_answers(df, args)

    
    return df


