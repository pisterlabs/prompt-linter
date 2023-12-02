# Load libraries
import os
import sys
import traceback
import time
import pandas as pd
import numpy as np
from datetime import datetime
import boto3
from dotenv import load_dotenv
import openai
path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
os.chdir(path)
sys.path.insert(0, path)
from lib.utils import get_conn, get_embeddings, execute_sql, categorize, ensure_columns, serialize_vector

today = datetime.today().strftime("%Y-%m-%d")
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
default_none_date = datetime.strptime('2260-12-31', "%Y-%m-%d")

##### Dependencies:  E_ideas_form_field_answers.py > T_param_class_table.py > L_param_class_table.py ###############


###############################ENVIRONMENT VARIABLES#####################################
load_dotenv()
aws_host = os.environ.get('aws_host')
aws_db = os.environ.get('aws_db')
aws_db_dw = os.environ.get('aws_db_dw')
aws_access_id = os.environ.get('aws_access_id')
aws_access_key = os.environ.get('aws_access_key')
aws_port = int(os.environ.get('aws_port'))
aws_user_db = os.environ.get('aws_user_db')
aws_pass_db = os.environ.get('aws_pass_db')
bucket_name = os.environ.get('bucket_name')


######################## AUXILIARY DICTIONARIES #########################################

f_ans_dict = {'Idea name_embedded':'name_embedded','Solution 1':'solution_1', 'Solution 2':'solution_2', 
        'Solution 3':'solution_3', 'Solution 4':'solution_4', 'Solution 5':'solution_5', 'Solution 6':'solution_6', 
        'Problem 1':'problem_1', 'Problem 2':'problem_2', 'Problem 3':'problem_3', 'Problem 4':'problem_4', 
        'Problem 5':'problem_5', 'Problem 6':'problem_6', 'Solution 1_embedded':'sol_1_embedded',
        'Solution 2_embedded':'sol_2_embedded', 'Solution 3_embedded':'sol_3_embedded',
        'Solution 4_embedded':'sol_4_embedded', 'Solution 5_embedded':'sol_5_embedded', 
        'Solution 6_embedded':'sol_6_embedded', 'Problem 1_embedded':'prob_1_embedded',
        'Problem 2_embedded':'prob_2_embedded', 'Problem 3_embedded':'prob_3_embedded',
        'Problem 4_embedded':'prob_4_embedded', 'Problem 5_embedded':'prob_5_embedded',
        'Problem 6_embedded':'prob_6_embedded'}

#Nested dictionary for company forms
conn1 = get_conn(aws_host, aws_db_dw, aws_port, aws_user_db, aws_pass_db)
query1 = "Select company_id, name, category from public.param_class_table"
result1 =  execute_sql(query1, conn1)

class_dict = {}

df_d = pd.DataFrame(result1, columns=['company_id', 'name', 'category'])   

for index, row in df_d.iterrows():
    company_id = row['company_id']
    name = row['name']
    category = row['category']

    if company_id not in class_dict:
        class_dict[company_id] = {}

    class_dict[company_id][name] = category

#Create dictionary from DIM_COMPANY table
conn2 = get_conn(aws_host, aws_db_dw, aws_port, aws_user_db, aws_pass_db)
query2 = "Select id, company_db_id from public.dim_company"
result2 = execute_sql(query2, conn2)

comp_dic = pd.DataFrame(result2, columns=['id', 'company_db_id']).set_index('company_db_id')['id'].to_dict()   

#Create dictionary from DIM_USERS table
conn3 = get_conn(aws_host, aws_db_dw, aws_port, aws_user_db, aws_pass_db)
query3 = "Select id, user_db_id from public.dim_users"
result3 = execute_sql(query3, conn3)
user_dic = pd.DataFrame(result3, columns=['id', 'user_db_id']).set_index('user_db_id')['id'].to_dict() 

#Create dictionary from DIM_IDEAS table

conn4 = get_conn(aws_host, aws_db_dw, aws_port, aws_user_db, aws_pass_db)
query4 = "Select id, idea_db_id from public.dim_idea"
result4 = execute_sql(query4, conn4)
idea_dic = pd.DataFrame(result4, columns=['id', 'idea_db_id']).set_index('idea_db_id')['id'].to_dict()

#Create dictionary from DIM_GOAL table
conn5 = get_conn(aws_host, aws_db_dw, aws_port, aws_user_db, aws_pass_db)
query5 = "Select id, goal_db_id from public.dim_goals"
result5 = execute_sql(query5, conn5)
goal_dic = pd.DataFrame(result5, columns=['id', 'goal_db_id']).set_index('goal_db_id')['id'].to_dict()

#Create dictionary from param_class_table

#################################AUXILIARY FUNCTIONS############################################


def process_data(df: pd.DataFrame, start_index=0) -> pd.DataFrame:
    """Function that processes the data frame and returns a new data frame with the embeddings"""
    length = len(df)
    MAX_RETRIES = 15

    for i, row in df.iterrows():
        index = i + start_index
        print(index)
        if (index-1) >= length:
            return df
        
        if row['field_answer'] == '':
            continue

        retries = 0
        success = False
        
        if row['field_answer'] is None or row['field_answer'] in ('None', ''):
            continue
        
        while retries < MAX_RETRIES and not success:
            try:
                embedding = get_embeddings(row['field_answer'])
                df.at[index, 'ada_embedded'] = np.array(embedding).tolist()
                success = True  # Mark as successful to exit the while loop
                time.sleep(1)  # Adjust based on your rate limit

            except openai.error.APIConnectionError:
                print(traceback.format_exc())
                print(f'Error in index: {index}. Skipping to next entry.')
                retries += 1
                backoff_time = 2 * retries  # geometrical backoff
                print(f'Error at index {index}. Retrying in {backoff_time} seconds...')
                time.sleep(backoff_time)

            except ValueError:
                print('ValueError, check if the final index is correct. Final index:', index)
                retries += 1
                backoff_time = 2 * retries  # geometrical backoff
                print(f'Error at index {index}. Retrying in {backoff_time} seconds...')
                time.sleep(backoff_time)
                
            except Exception as e:
                retries += 1
                backoff_time = 2 * retries  # geometrical backoff
                print(f'Error at index {index}. Retrying in {backoff_time} seconds...')
                time.sleep(backoff_time)

        
        if retries >= MAX_RETRIES:
            print(f"Maximum retries reached for index {index}. Skipping to next entry.")

    return df

def fill_prev_columns(row:pd.Series)->pd.Series:
    """Function that fills the previous columns with the last non-nan value

    Args:
        row (_type_): row of a data frame

    Returns:
        row: modify row
    """
    
    problem_columns = [f"problem_{i+1}" for i in range(6)]
    solution_columns = [f"solution_{i+1}" for i in range(6)]
    prob_emb_columns = [f"prob_{i+1}_embedded" for i in range(6)]
    sol_emb_columns = [f"sol_{i+1}_embedded" for i in range(6)]

    for columns in [problem_columns, solution_columns, prob_emb_columns, sol_emb_columns]:
        non_empty_cols = []
        for col in columns:
            value = row[col]
            try:
                if not pd.isnull(value):
                    non_empty_cols.append(col)
            except ValueError:
                non_empty_cols.append(col)
        
        # Shift values to the left
        for i, col in enumerate(columns):
            row[col] = row[non_empty_cols[i]] if i < len(non_empty_cols) else np.nan

    return row




#################################MAIN FUNCTION############################################

def main(url):

    if url == 'No new data to load.':
        print('No new data to load.' + ' ' + 'No new data to load.')
        sys.stdout.write('No new data to load.' + ' ' + 'No new data to load.')
        return 'No new data to load.' + ' ' + 'No new data to load.'
    
    s3_client = boto3.client('s3',
                        aws_access_key_id=aws_access_id,
                        aws_secret_access_key=aws_access_key)

    
    s3_file_name =  url.split('.com/')[-1]
    
    response = s3_client.get_object(Bucket=bucket_name, Key=s3_file_name)
    
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    
    if status == 200:
        print(f"Successful S3 get_object response. Status - {status}")
        df = pd.read_json(response.get("Body"))
        df = df.loc[24300:26200]

    else:
        print(f"Unsuccessful S3 get_object response. Status - {status}")
        raise Exception(f"Error loading data from S3 - {status}")
    
    #Classify form field answers from database
    df_ = df.loc[df['company_id'].isin(class_dict.keys())]
    df_.loc[:,'category'] = df_.apply(lambda x: categorize(x['title'], class_dict[x['company_id']]), axis=1)    
    
    #Get embeddeds from database
    df_emb = df_.loc[~df_['category'].isin(['Other','None', 'Category: Other']),].copy().reset_index(drop=True)
    df_emb['ada_embedded'] = None
    df_emb = process_data(df_emb)
    
    df_emb.rename(columns={'field_answer':'description_field', 'title':'idea_title'}, inplace=True)
    
    #Get main idea table and tag_table to merge it with the embeddeds
    
    q1 = """
    SELECT DISTINCT
        tags.name AS "tag_name",
        tags.description AS "tag_description",
        tags.id AS "tag_id",
        tags.tag_type AS "tag_type",
        ideas_tags.idea_id AS "idea_id",
        ideas.user_id AS "user_id",
        ideas.goal_id AS "goal_id",
        ideas.is_private AS "is_private",
        ideas.stage AS "stage",
        ideas.like_ideas_count AS "like_ideas_count",
        ideas.average_general AS "average_general",
        ideas.created_at AS "created_at",
        ideas.updated_at AS "updated_at",
        ideas.submited_at AS "submited_at"
    FROM
        public.tags
    LEFT JOIN
        public.ideas_tags ON tags.id = ideas_tags.tag_id
    LEFT JOIN
        public.ideas ON ideas.id = ideas_tags.idea_id;"""
    
    conn6 = get_conn(aws_host, aws_db, aws_port, aws_user_db, aws_pass_db)
    result6 = execute_sql(q1, conn6)
    df_tag = pd.DataFrame(result6, columns=['tag_name', 'tag_description', 'tag_id', 'tag_type', 'idea_id',
                                            'user_id', 'goal_id', 'is_private', 'stage', 'like_ideas_count', 'average_general', 'created_at',
                                            'updated_at', 'submited_at'])
    

    df_stg = pd.merge(df_emb, df_tag, how='left', on='idea_id')
    df_stg.rename(columns={'idea_id':'idea_db_id'}, inplace=True)

    stg_columns = ['idea_db_id', 'company_id', 'user_id', 'goal_id', 'tag_name', 'tag_description', 'tag_type', 'is_private', 
                'stage', 'like_ideas_count', 'average_general', 'idea_name', 'description_field', 
                'category','created_at', 'updated_at', 'submited_at','ada_embedded']
    df_stg = df_stg[stg_columns]
    
    # Then, apply the function to create new columns
    for category in df_stg['category'].unique():
        df_stg[category] = df_stg.loc[df_stg['category'] == category, 'description_field']
        df_stg[str(category)+'_embedded'] = df_stg.loc[df_stg['category'] == category, 'ada_embedded']
        

    df_grouped = df_stg.groupby('idea_db_id').agg(lambda x: np.nan if x.isna().all() else x.dropna().iloc[0])
    df_grouped.rename(columns=f_ans_dict, inplace=True)
    df_grouped = ensure_columns(df_grouped, list(f_ans_dict.values()))
    df_grouped.replace([None, '', np.nan], None, inplace=True)
    df_grouped.reset_index(inplace=True)
    df_grouped.rename(columns={'description_field':'description', 'company_id_x':'company_id', 
                            'category_y':'category', 'idea_name':'name'}, inplace=True)
    
    
    final_columns =['idea_db_id', 'company_id', 'user_id','goal_id', 'tag_name', 'tag_description', 'tag_type', 'is_private', 
                'stage', 'like_ideas_count', 'average_general', 'name', 'description', 
                'category','created_at', 'updated_at', 'submited_at','name_embedded', 'solution_1', 'solution_2',
                'solution_3', 'solution_4', 'solution_5', 'solution_6', 'problem_1', 'problem_2', 'problem_3', 'problem_4',
                'problem_5', 'problem_6', 'sol_1_embedded', 'sol_2_embedded', 'sol_3_embedded', 'sol_4_embedded',
                'sol_5_embedded', 'sol_6_embedded', 'prob_1_embedded', 'prob_2_embedded', 'prob_3_embedded',
                'prob_4_embedded', 'prob_5_embedded', 'prob_6_embedded']
    
    #Get final DF before splitting DIM table and FACT table. Fill the values of previous numbered columns, 
    # so that problem_1 or solution_1 are filled before problem_2 or solution_2 and so on. 
    df_final = df_grouped[final_columns]
    df_final = df_final.apply(fill_prev_columns, axis=1)
    df_final['valid_from'] = now
    df_final['valid_to'] = default_none_date
    df_final['is_current'] = True
    df_final['created_at'] = df_final['created_at'].replace({pd.NaT: default_none_date, 'None':default_none_date, '':default_none_date})
    df_final['updated_at'] = df_final['updated_at'].replace({pd.NaT: default_none_date, 'None':default_none_date, '':default_none_date})
    
    final_cols = ['idea_db_id', 'tag_name','tag_description', 'tag_type',
                    'is_private','category', 'stage', 'like_ideas_count', 
                    'average_general', 'name', 'description', 'problem_1', 'solution_1', 'problem_2', 
                    'solution_2', 'problem_3', 'solution_3', 'problem_4', 'solution_4', 'problem_5', 
                    'solution_5', 'problem_6', 'solution_6', 'name_embedded','prob_1_embedded', 'sol_1_embedded',
                    'prob_2_embedded', 'sol_2_embedded', 'prob_3_embedded', 'sol_3_embedded','prob_4_embedded',
                    'sol_4_embedded', 'prob_5_embedded', 'sol_5_embedded','prob_6_embedded', 'sol_6_embedded', 
                    'created_at', 'updated_at', 'valid_from', 'valid_to', 'is_current']
    df_final_dim_idea = df_final[final_cols]
    
    #Serialize embedded vector
    emb_cols = ['name_embedded','prob_1_embedded', 'sol_1_embedded',
                    'prob_2_embedded', 'sol_2_embedded', 'prob_3_embedded', 
                    'sol_3_embedded','prob_4_embedded', 'sol_4_embedded', 
                    'prob_5_embedded', 'sol_5_embedded','prob_6_embedded', 
                    'sol_6_embedded']
    
    for col in emb_cols:
        df_final_dim_idea[col] = df_final_dim_idea[col].apply(serialize_vector)
    

    #Dataframe split between DIM table and FACT table    
    df_final_fact_sub_idea = df_final[['idea_db_id', 'company_id', 'goal_id', 'submited_at']]
    df_final_fact_sub_idea.replace({pd.NaT: None, 'None':None}, inplace=True)
    df_final_fact_sub_idea.loc[:,'company_id'] = df_final_fact_sub_idea.loc[:,'company_id'].apply(lambda x: categorize(x, comp_dic))
    df_final_fact_sub_idea[['company_id']] = df_final_fact_sub_idea[['company_id']].replace('None', None)
    df_final_fact_sub_idea['goal_id'] = df_final_fact_sub_idea['goal_id'].apply(lambda x: categorize(x, goal_dic)).replace('None', None)
    df_final_fact_sub_idea = df_final_fact_sub_idea.replace({None: 0})
    df_final_fact_sub_idea['submited_at'] = df_final_fact_sub_idea['submited_at'].replace(0, default_none_date)
    
    #Upload to S3
    stage_file_dim = df_final_dim_idea.to_parquet()
    stage_file_fact = df_final_fact_sub_idea.to_parquet()
    
    s3_file_name_stg_dim = 'stage/' + str(today) + '_DIM_ideas.parquet'
    s3_file_name_stg_fact = 'stage/' + str(today) + '_FACT_submitted_ideas.parquet'
    
    s3_client.put_object(Bucket=bucket_name, Key=s3_file_name_stg_dim, Body=stage_file_dim)
    s3_client.put_object(Bucket=bucket_name, Key=s3_file_name_stg_fact, Body=stage_file_fact)
    
    
    url_dim = f'https://{bucket_name}.s3.amazonaws.com/{s3_file_name_stg_dim}'
    url_fact = f'https://{bucket_name}.s3.amazonaws.com/{s3_file_name_stg_fact}'
    
    response = s3_client.delete_object(Bucket=bucket_name, Key=s3_file_name)
    
    print(url_dim, url_fact)
    
    sys.stdout.write(url_dim + ' ' + url_fact)
    
    return url_dim + ' ' + url_fact
    
    
if __name__ == '__main__':
    url = sys.argv[1]
    main(url)