import openai
from dotenv import load_dotenv
import os
import pandas as pd
import ast
import time
from openai.error import APIError
path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
os.chdir(path)

load_dotenv()
path_to_drive = os.environ.get('path_to_drive')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY_INNK') 
OPENAI_ORG_ID = os.environ.get('OPENAI_ORG_ID_INNK')
openai.api_key = OPENAI_API_KEY
openai.organization = OPENAI_ORG_ID



###########################AUXILIARY FUNCTIONS#################################################

def create_message(df:pd.DataFrame)->str:
    
    base_message = """Given the cluster of ideas below, return a Python dictionary. Structure the dictionary as follows:
    {
    'cluster_name': 'Provide a 8-word max cluster name',
    'cluster_description': 'Provide a 25-word max cluster description',
    '1': {
        'name': 'Provide a 8-word max name for Idea 1',
        'description': 'Provide a 35-word max description for Idea 1'
    }
    }
    Ensure all answers are in this format. Below is the cluster:
    \n"""
        
    cluster_text = ''
    
    end_message = """Consider this correct example:
    {
    'cluster_name': 'Oferta Turística Comercial',
    'cluster_description': 'Ideas para potenciar turismo y comercio en estaciones',
    '214': {
        'name': 'Módulos Comerciales Plazoleta',
        'description': 'Proponer módulos para guías turísticos en la plazoleta de San Javier.'
    }
    }

    Avoid mistakes like this:
    {
    'cluster_name': 'Oferta Turística',
    'cluster_description': 'música',
    'ideas': {
        '214': {
        'name': 'Nombre',
        'description': 'Descripción'
        }
    }
    }

    Ensure to return ONLY the dictionary, maintaining the given idea number as the key for each idea. Never use the given idea name or description as your answer,
    Your response must always be in Spanish in perfect json format."""
    
    for i, row in df.iterrows():
        text = '[Idea number ' + str(i) + ': Name: ' + str(df.loc[i,'name']) + '| Description: ' + str(df.loc[i,'description']) + "]\n"
        if len(text) >1000:
            text = '[Idea number ' + str(i) + ': Name: ' + str(df.loc[i,'name']) + '| Description: ' + str(df.loc[i,'description'])[:500] + "]\n"
        cluster_text += text
    
    return clean_text(base_message + cluster_text + end_message)


def classify_cluster(df:pd.DataFrame)->dict:
    message = create_message(df)
    """
    response = openai.ChatCompletion.create(
            model ="gpt-3.5-turbo-instruct",
            messages = [{"role": "system", 
                          "content": "e a proficient assistant, experienced in categorizing and summarizing diverse \
                          ideas. You are tasked to succinctly summarize and classify the provided set of ideas. Please \
                              present your answer in a well-structured JSON format."}, 
                         {"role": "user", "content": message}],
            temperature=0,
    )
    """
    response = openai.Completion.create(
        model = "gpt-3.5-turbo-instruct",
        prompt = message,
        temperature=0,
        max_tokens=4000,
    )
    print(response['usage'])
    print(response.choices[0]['text'])
    #return ast.literal_eval(response.choices[0].message['content'])
    return ast.literal_eval(response.choices[0]['text'])



def fill_cluster(df:pd.DataFrame, answer:dict)->pd.DataFrame:
    """
    Fill the dataframe with the answers of the cluster classification

    Args:
        df (pd.DataFrame): Filtered dataframe of a cluster of ideas with its descritions
        answer (dict): dictionary with the answers of the cluster classification

    Returns:
        pd.DataFrame: dataframe with the answers of the cluster classification
    """
    df['cluster_name'] = answer['cluster_name']
    df['cluster_description'] = answer['cluster_description']
    for i, row in df.iterrows():
        df.loc[i,'idea_name'] = answer[str(i)]['name']
        df.loc[i,'idea_description'] = answer[str(i)]['description']
    return df


def chunk_dataframe(df, max_rows=5): 
    """
    Yields chunks of the dataframe with up to max_rows rows.
    """
    for i in range(0, len(df), max_rows):
        yield df.iloc[i:i + max_rows]

def classify_clusters_from_chunks(chunks):
    """
    Process dataframe chunks and classify them.
    """
    all_answers = {}
    
    for chunk in chunks:
        chunk =pd.DataFrame(chunk)
        answer = classify_cluster(chunk)
        all_answers.update(answer)
        
    return all_answers

def clean_text(text):
    # Remove extra lines and strip spaces from each line
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # Join the cleaned lines
    cleaned_text = ' '.join(lines)
    # Remove extra spaces
    cleaned_text = ' '.join(cleaned_text.split())
    return cleaned_text


df =  pd.read_excel(path_to_drive + r'cluster_117.xlsx')#dataframe with idea name, idea_id, goal_id, name and embedded column with a mean of all the embedded columns of the idea
df.rename(columns={'solution_1':'description', 'combined_labels':'cluster'}, inplace=True)
df.sort_values('cluster', inplace=True)
df = df.reset_index(drop=True)

def main():

    df_ = pd.DataFrame()
    n = 0
    for i in df['cluster'].unique():
        
        chunks = list(chunk_dataframe(df.loc[df.loc[:,'cluster']==i]))

        # Classify each chunk

        answers = classify_clusters_from_chunks(chunks)


        # Fill dataframe with answers
        df_stg = fill_cluster(df.loc[df.loc[:,'cluster']==i], answers)
        

        df_ = pd.concat([df_,df_stg])
        n += 1
        if n==7:
            break
        time.sleep(1)

if __name__ == '__main__':
    main()  