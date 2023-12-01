# Load libraries
import os
import sys
import traceback
import time
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import openai
path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
os.chdir(path)
sys.path.insert(0, path)
from lib.general_module import get_conn, categorize, execute_sql

today = datetime.today()#.strftime("%d-%m-%Y")
now = datetime.now()#.strftime("%d-%m-%Y, %H:%M:%S")

##### Dependencies: E_goal.py  ###############


###############################ENVIRONMENT VARIABLES#####################################
load_dotenv()
aws_host = os.environ.get('aws_host')
aws_db = os.environ.get('aws_db')
aws_db_dw = os.environ.get('aws_db_dw')
aws_port = int(os.environ.get('aws_port'))
aws_user_db = os.environ.get('aws_user_db')
aws_pass_db = os.environ.get('aws_pass_db')
path_to_drive = os.environ.get('path_to_drive')

######################## AUXILIARY DICTIONARIES #########################################

#Create dictionary from DIM_COMPANY table
conn1 = get_conn(aws_host, aws_db_dw, aws_port, aws_user_db, aws_pass_db)
query1 = "Select id, company_db_id from public.dim_company"
result1 = execute_sql(query1, conn1)
comp_dic = pd.DataFrame(result1, columns=['id', 'company_db_id']).set_index('company_db_id')['id'].to_dict()   



final_columns = ['goal_db_id', 'company_id', 'goal_name', 'goal_description', 'active',
                 'ideas_reception', 'is_private', 'end_campaign', 'created_at', 'updated_at', 
                 'valid_from', 'valid_to', 'is_current']
path_to_file = path_to_drive + r'/raw/goal.json'
df_goal = pd.read_json(path_to_file)
df_goal.rename(columns={'id': 'goal_db_id', 'name': 'goal_name', 'description': 'goal_description'}, inplace=True)
df_goal['company_id'] = df_goal['company_id'].apply(lambda x: categorize(x, comp_dic))
df_goal['valid_from'] = today
df_goal['valid_to'] = '9999-12-31'
df_goal['is_current'] = True
df_goal['company_id'] = df_goal['company_id'].replace('None', None)
df_goal = df_goal[final_columns]

df_goal.to_parquet(path_to_drive + r'stage/dim_goal.parquet', index=False)

