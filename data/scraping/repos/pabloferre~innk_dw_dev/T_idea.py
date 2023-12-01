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
from lib.general_module import get_conn, get_embeddings, ensure_columns, categorize, execute_sql, serialize_vector

today = datetime.today()#.strftime("%d-%m-%Y")
now = datetime.now()#.strftime("%d-%m-%Y, %H:%M:%S")

##### Dependencies: E_ideas_form_field_answers.py  ###############


###############################ENVIRONMENT VARIABLES#####################################
load_dotenv()
aws_host = os.environ.get('aws_host')
aws_db = os.environ.get('aws_db')
aws_db_dw = os.environ.get('aws_db_dw')
aws_db = os.environ.get('aws_db')
aws_port = int(os.environ.get('aws_port'))
aws_user_db = os.environ.get('aws_user_db')
aws_pass_db = os.environ.get('aws_pass_db')
path_to_drive = os.environ.get('path_to_drive')




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
query1 = "Select company_id, name, category from innk_dw_dev.public.param_class_table"
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
query2 = "Select id, company_db_id from innk_dw_dev.public.dim_company"
result2 = execute_sql(query2, conn2)

comp_dic = pd.DataFrame(result2, columns=['id', 'company_db_id']).set_index('company_db_id')['id'].to_dict()   

#Create dictionary from DIM_USERS table
conn3 = get_conn(aws_host, aws_db_dw, aws_port, aws_user_db, aws_pass_db)
query3 = "Select id, user_db_id from innk_dw_dev.public.dim_users"
result3 = execute_sql(query3, conn3)
user_dic = pd.DataFrame(result3, columns=['id', 'user_db_id']).set_index('user_db_id')['id'].to_dict() 

#Create dictionary from DIM_IDEAS table

conn4 = get_conn(aws_host, aws_db_dw, aws_port, aws_user_db, aws_pass_db)
query4 = "Select id, idea_db_id from innk_dw_dev.public.dim_idea"
result4 = execute_sql(query4, conn4)
idea_dic = pd.DataFrame(result4, columns=['id', 'idea_db_id']).set_index('idea_db_id')['id'].to_dict()

#Create dictionary from DIM_GOAL table
conn5 = get_conn(aws_host, aws_db_dw, aws_port, aws_user_db, aws_pass_db)
query5 = "Select id, goal_db_id from innk_dw_dev.public.dim_goal"
result5 = execute_sql(query5, conn5)
goal_dic = pd.DataFrame(result5, columns=['id', 'goal_db_id']).set_index('goal_db_id')['id'].to_dict()


#################################AUXILIARY FUNCTIONS############################################


def process_data(df: pd.DataFrame, start_index=0) -> pd.DataFrame:
    """Function that processes the data frame and returns a new data frame with the embeddings"""
    length = len(df)
    MAX_RETRIES = 25

    for i, row in df.iterrows():
        index = i + start_index
        print(index)
        if (index-1) >= length:
            return df

        retries = 0
        success = False

        while retries < MAX_RETRIES and not success:
            try:
                embedding = get_embeddings(row['field_answer'])
                df.at[index, 'ada_embedded'] = np.array(embedding).tolist()
                success = True  # Mark as successful to exit the while loop
                time.sleep(1)  # Adjust based on your rate limit

            except openai.error.APIConnectionError:
                print(traceback.format_exc())
                print(f'Error in index: {index}. Skipping to next entry.')
                return df

            except ValueError:
                print('ValueError, check if the final index is correct. Final index:', index)
                return df
                
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

def main():
    #Classify form field answers from database
    df_forms_ans = pd.read_excel(path_to_drive + r'raw/ideas_form_field_answers.xlsx')
    df_forms_ans = df_forms_ans.loc[df_forms_ans['company_id'].isin(class_dict.keys())]
    df_forms_ans['category'] = df_forms_ans.apply(lambda x: categorize(x['title'], class_dict[x['company_id']]), axis=1)

    #Get embeddeds from database
    df_emb = df_forms_ans.loc[~df_forms_ans['category'].isin(['Other','None']),].copy().reset_index(drop=True)
    df_emb['ada_embedded'] = None
    df_emb = process_data(df_emb)
    df_emb.rename(columns={'field_answer':'description_field'}, inplace=True)
    df_emb.to_parquet(path_to_drive + r'temp_ideas_embedded.parquet') #Temp file to save the embeddeds
    
    #Get main idea table and tag_table to merge it with the embeddeds

    df_idea = pd.read_json(path_to_drive + r'raw/ideas_table.json')
    df_idea.rename(columns={'title':'idea_title'}, inplace=True)

    df_ideas_tags = pd.read_json(path_to_drive + r'raw/ideas_tags_table.json')
    df_ideas_tags.rename(columns={'name':'tag_name', 'description':'tag_description'}, inplace=True)

    df_stg = pd.merge(df_idea, df_emb, how='right', right_on='idea_id', left_on='id')
    df_stg = pd.merge(df_stg, df_ideas_tags, how='left', on='idea_id')
    df_stg.rename(columns={'id':'idea_db_id'}, inplace=True)

    stg_columns = ['idea_db_id', 'company_id_x', 'user_id', 'goal_id', 'tag_name', 'tag_description', 'tag_type', 'is_private', 
                    'stage', 'like_ideas_count', 'average_general', 'idea_title', 'description_field', 
                    'category_y','created_at', 'updated_at', 'submited_at','ada_embedded']
    df_stg = df_stg[stg_columns]

    # Then, apply the function to create new columns
    for category in df_stg['category_y'].unique():
        df_stg[category] = df_stg.loc[df_stg['category_y'] == category, 'description_field']
        df_stg[str(category)+'_embedded'] = df_stg.loc[df_stg['category_y'] == category, 'ada_embedded']

    final_columns =['idea_db_id', 'company_id', 'user_id','goal_id', 'tag_name', 'tag_description', 'tag_type', 'is_private', 
                    'stage', 'like_ideas_count', 'average_general', 'name', 'description', 
                    'category','created_at', 'updated_at', 'submited_at','name_embedded', 'solution_1', 'solution_2',
                    'solution_3', 'solution_4', 'solution_5', 'solution_6', 'problem_1', 'problem_2', 'problem_3', 'problem_4',
                    'problem_5', 'problem_6', 'sol_1_embedded', 'sol_2_embedded', 'sol_3_embedded', 'sol_4_embedded',
                    'sol_5_embedded', 'sol_6_embedded', 'prob_1_embedded', 'prob_2_embedded', 'prob_3_embedded',
                    'prob_4_embedded', 'prob_5_embedded', 'prob_6_embedded']

    df_grouped = df_stg.groupby('idea_db_id').agg(lambda x: np.nan if x.isna().all() else x.dropna().iloc[0])
    df_grouped.rename(columns=f_ans_dict, inplace=True)
    df_grouped = ensure_columns(df_grouped, list(f_ans_dict.values()))
    df_grouped.replace([None, ''], np.nan, inplace=True)
    df_grouped.reset_index(inplace=True)
    df_grouped.rename(columns={'description_field':'description', 'company_id_x':'company_id', 
                            'category_y':'category', 'Idea name':'name'}, inplace=True)
    
    #Get final DF before splitting DIM table and FACT table. Fill the values of previous numbered columns, 
    # so that problem_1 or solution_1 are filled before problem_2 or solution_2 and so on. 
    df_final = df_grouped[final_columns]
    df_final = df_final.apply(fill_prev_columns, axis=1)
    df_final['valid_from'] = today
    df_final['valid_to'] = '9999-12-31'
    df_final['is_current'] = True
    final_cols = ['idea_db_id', 'tag_name','tag_description', 'tag_type',
                    'is_private','category', 'stage', 'like_ideas_count', 
                    'average_general', 'name', 'description', 'problem_1', 'solution_1', 'problem_2', 
                    'solution_2', 'problem_3', 'solution_3', 'problem_4', 'solution_4', 'problem_5', 
                    'solution_5', 'problem_6', 'solution_6', 'name_embedded','prob_1_embedded', 'sol_1_embedded',
                    'prob_2_embedded', 'sol_2_embedded', 'prob_3_embedded', 'sol_3_embedded','prob_4_embedded',
                    'sol_4_embedded', 'prob_5_embedded', 'sol_5_embedded','prob_6_embedded', 'sol_6_embedded', 
                    'created_at', 'updated_at', 'valid_from', 'valid_to', 'is_current']
    df_final_dim_idea = df_final[final_cols]
    emb_cols = ['name_embedded','prob_1_embedded', 'sol_1_embedded',
                    'prob_2_embedded', 'sol_2_embedded', 'prob_3_embedded', 
                    'sol_3_embedded','prob_4_embedded', 'sol_4_embedded', 
                    'prob_5_embedded', 'sol_5_embedded','prob_6_embedded', 
                    'sol_6_embedded']
    

    for col in emb_cols:
        df_final_dim_idea[col] = df_final_dim_idea[col].apply(serialize_vector)
        
    df_final_fact_sub_idea = df_final[['idea_db_id', 'company_id', 'goal_id', 'submited_at']]
    df_final_fact_sub_idea.replace({pd.NaT: None, 'None':None}, inplace=True)
    df_final_fact_sub_idea.loc[:,'company_id'] = df_final_fact_sub_idea.loc[:,'company_id'].apply(lambda x: categorize(x, comp_dic))
    df_final_fact_sub_idea[['company_id']] = df_final_fact_sub_idea[['company_id']].replace('None', None)
    df_final_fact_sub_idea['goal_id'] = df_final_fact_sub_idea['goal_id'].apply(lambda x: categorize(x, goal_dic)).replace('None', None)
    df_final_dim_idea.to_parquet(path_to_drive + r'stage/dim_idea.parquet', index=False)
    df_final_fact_sub_idea.to_parquet(path_to_drive + r'raw/fact_sub_idea.parquet', index=False)
    
    return None

if __name__ == '__main__':
    main() 