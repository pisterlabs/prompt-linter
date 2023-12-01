import openai
from dotenv import load_dotenv
import os
import pandas as pd
import numpy as np
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

from langchain.chat_models import ChatOpenAI
from langchain.chains import create_tagging_chain, create_tagging_chain_pydantic, create_extraction_chain_pydantic
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.prompts.example_selector.base import BaseExampleSelector
from langchain.callbacks import get_openai_callback
from langchain.llms import OpenAI
from pydantic import BaseModel, Field
from typing import Dict, List, Sequence



#################################################CLASSES ####################################################
class CustomExampleSelector(BaseExampleSelector):
    
    def __init__(self, examples: List[Dict[str, str]]):
        self.examples = examples
    
    def add_example(self, example: Dict[str, str]) -> None:
        """Add new example to store for a key."""
        self.examples.append(example)

    def select_examples(self, input_variables: Dict[str, str]) -> List[dict]:
        """Select which examples to use based on the inputs."""
        return np.random.choice(self.examples, size=2, replace=False)
    

class Idea(BaseModel):
    #id: int = Field(description='Given Idea Number for the idea')
    idea_name: str = Field(description='Name of the idea in no more than 5 words')
    idea_description: str = Field(description='New summarized description of the idea in no more than 35 words')
    

class Cluster(BaseModel):
    cluster_name: str = Field(description='Name of the cluster in no more than 5 words')
    cluster_description: str = Field(description='Summarized description of the cluster in no more than 35 words')
    idea: Sequence[Idea]
    
class Cluster_(BaseModel):
    cluster_name: str = Field(description='Name of the cluster in no more than 5 words')
    cluster_description: str = Field(description='Summarized escription of the cluster in no more than 35 words')



#################################################PROMPT TEMPLATE##############################################
system_template = "You are a proficient assistant, experienced in categorizing and summarizing diverse ideas.\
    You are tasked to succinctly summarize and classify the provided set of ideas. Please present your answer in \
        a well-structured JSON format."
 
template = """For each idea in the following list, please provide a refreshed idea name (up to 8 words), and an idea \
    description with a concise summary (from 10 words and up to 35 words). Also, return a collective cluster name (up to 8 words) and \
        description (form 10 words and up to 30 words).
\n
List of Ideas: 
\n
{ideas}
\n
Note: Ensure the new idea names and summaries are delivered in Spanish. One response is required for each idea \
    listed. Don't copy the given idea name and idea description in the response."""

template_prev_chunk = """Provide a cluster name (max 8 words) and  cluster description (max 25 words). Consider \
    combining the cluster names and descriptions from chunks of ideas from the list given below.
\n
List of cluster names: {clu_name}
\n
List of cluster descriptions: {clu_description}
\n
Note: Ensure the new cluster names and summaries are delivered in Spanish."""



example = "{'cluster_name': 'Oferta turística y comercial', 'cluster_description': \
    'Ideas para mejorar la oferta turística y comercial en las estaciones', 'id':'0', 'idea_name': 'Módulos comerciales en plazoleta', \
    'idea_description': 'Implementar módulos para guías turísticos en la plazoleta de la estación San Javier.', 'id':'1', 'idea_name': 'Módulos comerciales', \
    'idea_description': 'Implementar módulos para aviones.'}"


llm = ChatOpenAI(temperature=0,
                 model='gpt-3.5-turbo-16k',
                 openai_api_key=OPENAI_API_KEY
)
example_selector = CustomExampleSelector(example)
system_message_prompt = SystemMessagePromptTemplate.from_template(template=system_template)
#human_message_prompt = HumanMessagePromptTemplate.from_template(template=template, example_selector=example_selector)
#chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])


###########################AUXILIARY FUNCTIONS#################################################

def get_ai_idea_classification(df:pd.DataFrame, clust_dict)->pd.DataFrame:
    cluster_name = clust_dict['cluster_name']
    cluster_description = clust_dict['cluster_description'] 
    cluster_text = df[['name', 'description']].to_dict(orient='records')
    i = 0
    for di in cluster_text:
        di['idea_number'] = str(i)
        di['idea_name'] = di.pop('name')
        di['idea_description'] = final_text(di.pop('description'))
        i += 1
    input_variables = {'ideas':cluster_text, 'cluster_name':cluster_name, 'cluster_description':cluster_description}
    llm = ChatOpenAI(temperature=0,
                    model='gpt-3.5-turbo-16k',
                    openai_api_key=OPENAI_API_KEY,
                    max_tokens=4000
)
    system_message_prompt = SystemMessagePromptTemplate.from_template(template=system_template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(template=template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
    chain = create_tagging_chain_pydantic(Cluster, llm, chat_prompt, verbose=True)
    with get_openai_callback() as cb:
        cluster_instance = chain.run(ideas=input_variables['ideas'], cluster_name=input_variables['cluster_name'], cluster_description=input_variables['cluster_description'])
        #print(cb)
    
    
    c_name = cluster_instance.cluster_name
    c_description = cluster_instance.cluster_description
    if clust_dict['flag']:
        clu_name = [clust_dict['cluster_name'], cluster_instance.cluster_name]
        clu_description = [clust_dict['cluster_description'] , cluster_instance.cluster_description]
        human_message_prompt = HumanMessagePromptTemplate.from_template(template=template_prev_chunk)
        chat_prompt2 = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])
        chain2 = create_tagging_chain_pydantic(Cluster_, llm, chat_prompt2, verbose=True)
        
        with get_openai_callback() as cb:
            cluster_instance2 = chain2.run(clu_name=clu_name, clu_description=clu_description)
            print(cb)
        c_name = cluster_instance2.cluster_name
        c_description = cluster_instance2.cluster_description
    flattened = [{"cluster_name": c_name, "cluster_description": c_description, **idea.dict()
    } for idea in cluster_instance.idea]

    # Create a DataFrame with one row for each idea
    return pd.DataFrame(flattened).rename(columns={'Idea_number':'id', 'Idea_name':'idea_name', 'Idea_description':'idea_description'})


def fill_cluster(df:pd.DataFrame, answer:pd.DataFrame)->pd.DataFrame:
    """
    Fill the dataframe with the answers of the cluster classification

    Args:
        df (pd.DataFrame): Filtered dataframe of a cluster of ideas with its descritions
        answer (dict): dictionary with the answers of the cluster classification

    Returns:
        pd.DataFrame: dataframe with the answers of the cluster classification
    """
    
    for i, row in df.iterrows():
        df.loc[i, 'cluster_name'] = answer.loc[i, 'cluster_name']
        df.loc[i, 'cluster_description'] = answer.loc[i, 'cluster_description']
        df.loc[i, 'idea_name'] = answer.loc[i,'idea_name']
        df.loc[i, 'idea_description'] = answer.loc[i,'idea_description']
    return df.reset_index(drop=True)


def chunk_dataframe(df, max_rows=5): 
    """
    Yields chunks of the dataframe with up to max_rows rows.
    """
    for i in range(0, len(df), max_rows):
        yield df.iloc[i:i + max_rows]

def classify_clusters_from_chunks(chunks:pd.DataFrame):
    """
    Process dataframe chunks and classify them.
    """
    all_answers = pd.DataFrame()
    clust_dict = {'cluster_name':'', 'cluster_description':'', 'flag':False}
    for chunk in chunks:
        chunk =pd.DataFrame(chunk)
        answer = get_ai_idea_classification(chunk, clust_dict)
        all_answers = pd.concat([all_answers, answer]).reset_index(drop=True)
        clust_dict['cluster_name'] = answer.loc[0, 'cluster_name']
        clust_dict['cluster_description'] = answer.loc[0, 'cluster_description']
        clust_dict['flag'] = True
        
    return all_answers.reset_index(drop=True)

def final_text(text):
    # Remove extra lines and strip spaces from each line
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # Join the cleaned lines
    cleaned_text = ' '.join(lines)
    # Remove extra spaces
    cleaned_text = ' '.join(cleaned_text.split())
    # Remove text len greater than 500
    if len(cleaned_text) > 500:
        cleaned_text = cleaned_text[:500]
    return cleaned_text


######################################################### LOAD DATA#################################################3

df =  pd.read_excel(path_to_drive + r'cluster_117.xlsx')#dataframe with idea name, idea_id, goal_id, name and embedded column with a mean of all the embedded columns of the idea
df.rename(columns={'solution_1':'description', 'combined_labels':'cluster'}, inplace=True)
df.sort_values('cluster', inplace=True)
df = df.reset_index(drop=True)



######################################################### MAIN #################################################

def main():

    df_ = pd.DataFrame()
    n = 0
    for i in df['cluster'].unique():
        
        chunks = list(chunk_dataframe(df.loc[df.loc[:,'cluster']==i]))

        if len(chunks)>1:
            memory = True
        
        # Classify each chunk

        answers = classify_clusters_from_chunks(chunks)
        

        # Fill dataframe with answers
        df_stg = fill_cluster(df.loc[df.loc[:,'cluster']==i].reset_index(drop=True), answers)
        

        df_ = pd.concat([df_,df_stg])
        if n==7:
            break
        time.sleep(4)
        n+=1

if __name__ == '__main__':
    main()  
    
    
