from db_session import * # custom .py file
import pandas as pd
import sys
import os
sys.path.append(r"C:\Users\silvh\OneDrive\lighthouse\Ginkgo coding\content-summarization\src")
from file_functions import *
# from response_processing import * # custom .py file
import time
import pytz
import re
from itertools import product
import openai
from prompts import * # custom .py file
import json

class Chaining:
    def __init__(self, text_id, title, text, folder_path, system_role="You are a helpful assistant.", 
            model="gpt-3.5-turbo", temperature=0.7, max_tokens=9000, 
        ):
        self.reference_id = text_id
        self.title = title
        self.text = text
        self.folder = re.sub(r'(?:.*\/)?(.*\/.*)\/?$', r'\1', folder_path)
        self.system_role = system_role
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model
        print(f'***OpenAI model: {self.model}')

    def create_prompt(self, task, text):
        system_role = f'{self.system_role}'
        user_input = f"""Given the following text delimited by triple backticks: ```{text}``` \n {task}"""
        messages = [
        {"role": "system", "content": system_role},
        {"role": "user", "content": user_input},]

        print('\tDone creating prompt')
        return messages

    def gpt(self, messages, n_choices, temperature, model=None):
        model = self.model if model == None else model
        print(f'\tSending request to {model}')
        print(f'\t\tRequesting {n_choices} choices using {model}')
        openai.api_key = os.getenv('api_key_openai')
        response = openai.ChatCompletion.create(
            model=model, messages=messages, 
            temperature=temperature, 
            max_tokens=self.max_tokens,
            n=n_choices
            )
        print('\tDone sending request to GPT-3')
        return response

    def summarize(
            self, task, prep_step, edit_task, simplify_task, simplify_audience,
            format_task,
            n_choices=5, task_first=True):
        if task_first == True:
            full_task = f'{task}\n\n{prep_step}\n\n{edit_task}\n\n{simplify_task} {simplify_audience}\n\n{format_task}'
        else:
            full_task = f'{prep_step}\n\n{task}\n\n{edit_task}\n\n{simplify_task} {simplify_audience}\n\n{format_task}'
        prompt = self.create_prompt(full_task, self.text)
        firstline_pattern = r'\s?(\S*)(\n*)(.+)'
        title = re.match(firstline_pattern, self.text)[0]
        self.qna = dict() 
        self.qna['timestamp'] = str(datetime.now(pytz.timezone('Canada/Pacific')))
        self.qna['reference_id'] = self.reference_id
        self.qna['article_title'] = self.title
        self.qna['text'] = self.text
        self.qna['system_role'] = self.system_role
        self.qna['model'] = self.model        
        self.qna['temperature'] = self.temperature
        self.qna['prep_step'] = prep_step.strip()
        self.qna['summarize_task'] = task.strip()
        self.qna['edit_task'] = edit_task.strip()
        self.qna['simplify_task'] = simplify_task.strip()
        self.qna['simplify_audience'] = simplify_audience.strip()
        self.qna['format_task'] = format_task.strip()
        self.qna['full_summarize_task'] = full_task.strip()
        self.qna['folder'] = self.folder
        self.summaries_dict = dict()
        self.article_title = title
        self.response_regex = r'response_(.*)'
        self.simple_summary_dict = dict()
        self.relevance_dict = dict()
        self.n_previous_prompts = dict()

        try:
            response = self.gpt(prompt, n_choices=n_choices, temperature=self.temperature)
        except Exception as error:
            exc_type, exc_obj, tb = sys.exc_info()
            f = tb.tb_frame
            lineno = tb.tb_lineno
            filename = f.f_code.co_filename
            print("An error occurred on line", lineno, "in", filename, ":", error)
            print('\t**API request failed for `.summarize()`**')
            return self.qna
        try:
            for index, choice in enumerate(response.choices):
                self.summaries_dict[f'response_{"{:02d}".format(index+1)}'] = choice["message"]["content"]
            self.qna.setdefault('summary', [])
            self.qna['summary'].extend([value for value in self.summaries_dict.values()])
        except Exception as error:
            exc_type, exc_obj, tb = sys.exc_info()
            f = tb.tb_frame
            lineno = tb.tb_lineno
            filename = f.f_code.co_filename
            print("An error occurred on line", lineno, "in", filename, ":", error)
            print('\t**Error with response parsing**')
    
def batch_summarize(sources_df, folder_path, prep_step, summarize_task, edit_task, 
    simplify_task, simplify_audience, format_task,
    chaining_bot_dict, iteration_id, task_first=True,
    system_role=None, model='gpt-3.5-turbo', max_tokens=1000, temperature=0.7, pause_per_request=0, n_choices=5,
    save_outputs=False
    ):
    prompts_df = pd.DataFrame(product(prep_step, summarize_task, edit_task, simplify_task, simplify_audience, format_task), 
        columns=['prep_step', 'summarize_task', 'edit_task', 'simplify_task', 'simplify_audience', 'format_task'])

    chaining_bot_dict[iteration_id] = dict()
    def summarize_from_df_row(text_id, title, text, chaining_bot_dict):
        for index in prompts_df.index:
            print(f'**Text #{text_id} prompt #{index+1} of {prompts_df.index.max()+1}**')
            task = prompts_df.loc[index, 'summarize_task']
            prep_step = prompts_df.loc[index, 'prep_step']
            edit_task = prompts_df.loc[index, 'edit_task']
            simplify_task = prompts_df.loc[index, 'simplify_task']
            simplify_audience = prompts_df.loc[index, 'simplify_audience']
            format_task = prompts_df.loc[index, 'format_task']
            try:
                print('Creating Chaining class instance')
                chatbot = Chaining(
                    text_id, title, text, folder_path=folder_path, system_role=system_role, 
                    model=model, max_tokens=max_tokens, temperature=temperature)
                print('Chaining class instance created')
                chatbot.summarize(
                    task=task, prep_step=prep_step, edit_task=edit_task, 
                    simplify_task=simplify_task, simplify_audience=simplify_audience,
                    format_task=format_task, n_choices=n_choices, task_first=task_first
                    )
                chaining_bot_dict[iteration_id][f'{text_id}_prompt{"{:02d}".format(index)}'] = chatbot
                print('\t...Completed')
                if pause_per_request > 0:
                    print(f'[batch_summarize()] Sleeping {pause_per_request} sec to avoid exceeding API rate limit')
                    time.sleep(pause_per_request) # Account for API rate limit of 3 API requests/limit 
            except Exception as error:
                exc_type, exc_obj, tb = sys.exc_info()
                f = tb.tb_frame
                lineno = tb.tb_lineno
                file = f.f_code.co_filename
                print("An error occurred on line", lineno, "in", file, ":", error)
                print('\t...Error making chatbot request')
                break
    sources_df.apply(lambda row: summarize_from_df_row(row['id'], row['title'], row['text'], chaining_bot_dict), axis=1)
    
    if save_outputs:
        try:
            save_instance_to_dict(
                chaining_bot_dict[iteration_id], 
                description=f'batch_Chaining_attributes_initial',
                ext=None, json_path=folder_path
                )
        except Exception as error:
            exc_type, exc_obj, tb = sys.exc_info()
            f = tb.tb_frame
            lineno = tb.tb_lineno
            file = f.f_code.co_filename
            print(f'An error occurred on line {lineno} in {file}: {error}')
            print('[batch_summarize_chain()] Unable to save API response')

    return chaining_bot_dict

def create_summaries_df(
    qna_dict, chatbot_dict, iteration_id, chatbot_id=None, 
    ):
    """
    Create DataFrame from initial ChatGPT summaries.

    Parameters:
        - qna_dict (dict): Dictionary for storing the DataFrames of the summaries.
        - chatbot_dict (dict): Dictionary of ChatGPT instances from the `batch_summarize` function in `orm_summarize.py`.
        - iteration_id (int): Iteration ID for content generation.
        - chatbot_id (int): Chatbot ID for content generation. Usually the same as the iteration_id.

    Returns:
        - qna_dict (dict): Dictionary of DataFrames of the summaries.
    """
    dfs_list = []
    chatbot_id = iteration_id if chatbot_id == None else chatbot_id
    for chatbot_key in chatbot_dict[chatbot_id].keys():
        print(f'Processing {chatbot_key}...')
        try:
            dfs_list.append(pd.DataFrame(
                chatbot_dict[chatbot_id][chatbot_key].qna, 
                index=[choice for choice in range(1, len(chatbot_dict[chatbot_id][chatbot_key].qna['summary'])+1)])
                )
        except Exception as error:
            exc_type, exc_obj, tb = sys.exc_info()
            file = tb.tb_frame
            lineno = tb.tb_lineno
            filename = file.f_code.co_filename
            print(f'\tAn error occurred on line {lineno} in {filename}: {error}')    
            print(f'Error creating DataFrame from {chatbot_key}: {error}')

    
    qna_df = pd.concat(dfs_list).reset_index(names=['choice'])
    qna_df = extract_summary(qna_df, 'summary')
    columns = qna_df.columns.tolist()
    columns.remove('choice')
    columns.insert(3, 'choice') # Move 'choice' column

    # qna_df['date'] = pd.Series('2023-06-12', index=qna_df.index)
    # columns.insert(0, 'date')


    qna_dict[iteration_id] = qna_df[columns]
    print(f'Original summaries DataFrame shape: {qna_df.shape}')
    print(f'\tOriginal summaries Dataframe columns: {qna_df.columns}')
    return qna_dict


def extract_summary(df, summary_column='summary'):
    """
    Convert the string in a DataFrame column to JSON.
    
    Parameters:
        - df (pandas DataFrame): DataFrame containing the column with a string in JSON format.
        - summary_column (str): Name of the column containing the string in JSON format to be converted.

    Returns:
        - df (pandas DataFrame): DataFrame with the JSON string column parsed into separate columns.

    """
    try:
        df[summary_column] = df[summary_column].apply(json.loads)
    except Exception as error:
        print(f'Error converting {summary_column} column to JSON: {error}; will do row by row')
        summary_list = []
        for index, summary in df[summary_column].items():
            try:
                summary_list.append(json.loads(summary))
            except Exception as error:
                print(f'Error converting summary {index} to JSON: {error}')
                summary_list.append(summary)
    def extract_value_from_key(summary, key):
        try:
            return summary[key]
        except Exception as error:
            match = re.search(rf'"{key}":\s*"([^"]+)"', summary)
            value = match.group(1) if match else None
            return value

    # Extract 'headline' and 'body' values
    df['headline'] = df[summary_column].apply(lambda x: extract_value_from_key(x, 'headline'))
    df['simple_summary'] = df[summary_column].apply(lambda x: extract_value_from_key(x, 'audience'))
    df[summary_column] = df[summary_column].apply(lambda x: extract_value_from_key(x, 'body'))
    df['simple_summary'] = df['simple_summary'].fillna(df[summary_column])

    return df

def openai_models(env="api_key_openai", query='gpt'):
    """
    List the availabel OpenAI models.
    Parameters:
        - env (str): Name of environmental variable storing the OpenAI API key.
        - query (str): Search term for filtering models.
    """
    openai.api_key = os.getenv(env)
    response = openai.Model.list()
    filtered_models = [model for model in response['data'] if model['id'].find(query) != -1]

    for item in filtered_models:
        print(item['id'])
    return filtered_models