#!/usr/bin/env python

from getpass import getpass
import os
import inspect

import time
import openai
from langchain.llms import OpenAI
from dotenv import load_dotenv
import tiktoken
import json
from pprint import pprint
from typing import Dict, List, Union, Optional, Tuple
import asyncio

import pandas as pd

# Load environment variables from .env file
load_dotenv()

# Get the OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY


# ## Template Draft 1*

system_prompt_1 = """You are an Expert in Python Code Analysis. Upon receiving Code snippets - Generate a concise, machine-readable PyDoc for the provided Python function;
-Focus on clear and straightforward descriptions without excessive detail, ensuring that parameter types, return types, and basic functionality are accurately described.
-Never Guess!
-If your are uncertain about parts of the code, write a request for clarification in the PyDocs: @RFMO: [your question}.
-If you recognize an opportunity for optimization in the code, mark it in the PyDocs:
--@CBOE = Can Be Optimized For Efficiency.
--@CBOR = Can Be Optimized For Readability.
--@CBOB = Can Be Optimized For Both Efficiency and Readability.
- Dont include the optimization in your response, Just the appropriate tag(s) please.
Do not include anything else in your response. \n"""

def list_files(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_paths.append(os.path.join(root, file))
    return file_paths

# Usage
directory = './funx'
all_files = list_files(directory)


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens

def input_cost_calculator(num_tokens):
    pricing_per_1K = {
        "GPT-4": {"8K_context": 0.03, "32K_context": 0.06},
        "GPT-3.5_Turbo": {"4K_context": 0.0015, "16K_context": 0.003},
        "FineTuningModels": {
            "babbage-002": {"training": 0.0004, "input_usage": 0.0016, "output_usage": 0.0016},
            "davinci-002": {"training": 0.0060, "input_usage": 0.0120, "output_usage": 0.0120},
            "GPT-3.5_Turbo": {"training": 0.0080, "input_usage": 0.0120, "output_usage": 0.0160}
        },
        "EmbeddingModels": {"Ada_v2": 0.0001},
        "BaseModels": {"babbage-002": 0.0004, "davinci-002": 0.0020}
    }

    cost_calculations = {}
    for model, contexts in pricing_per_1K.items():
        cost_calculations[model] = {}
        for context, price_per_1K in contexts.items():
            cost_calculations[model][context] = price_per_1K * (num_tokens / 1000)

    return cost_calculations

def read_files_into_dataframes(directory_path):
    dataframes = {'json': [], 'csv': []}
    
    # Walk through the directory structure
    for root, dirs, files in os.walk(directory_path):
        # Skip files in the 'errors' directory
        if 'errors' in root.split(os.sep):
            continue
            
        for file in files:
            # Check if the file is a .csv or .json file
            if file.endswith('.csv') or file.endswith('.json'):
                file_path = os.path.join(root, file)
                
                # Read the file into a dataframe
                if file.endswith('.csv'):
                    df = pd.read_csv(file_path)
                    dataframes['csv'].append(df)
                elif file.endswith('.json'):
                    df = pd.read_json(file_path)
                    dataframes['json'].append(df)
                
    return dataframes

data = read_files_into_dataframes('./funx')


def rename_columns_to_lowercase(data):
    for file_type, dfs in data.items():
        for df in dfs:
            df.columns = [col.lower() for col in df.columns]


# Rename all column names to lowercase
rename_columns_to_lowercase(data)


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding("cl100k_base")
    num_tokens = len(encoding.encode(string))
    return num_tokens
    
def num_tokens_from_file(file_path: str) -> int:
    """Returns the number of tokens in a text file."""
    encoding = tiktoken.get_encoding("cl100k_base")
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return num_tokens_from_string(text)

def num_tokens_from_dir(target_dir: str, encoding_name: Optional[str] = "cl100k_base") -> int:
    """Returns the total number of tokens in all text files in a directory."""
    total_tokens = 0
    for root, dirs, files in os.walk(target_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                total_tokens += num_tokens_from_file(file_path)
    return total_tokens

def num_tokens_from_dir_nested(
    target_dir: str, 
    encoding_name: Optional[str] = "cl100k_base",
    as_list: bool = False
) -> Union[Dict[str, int], List[Tuple[str, int]]]:
    """Returns a nested dictionary with token counts for each directory and subdirectory,
    or a list of tuples if as_list is True."""
    
    start_time = time.time()  # Record the start time
    
    result = {}
    for root, dirs, _ in os.walk(target_dir):
        total_tokens = 0
        nested_result = {}
        for dir in dirs:
            nested_dir = os.path.join(root, dir)
            nested_result[dir] = num_tokens_from_dir(nested_dir, encoding_name)
            total_tokens += nested_result[dir]
        result[root] = {'total_tokens': total_tokens, 'subdirs': nested_result}
    
    end_time = time.time()  # Record the end time
    total_runtime = end_time - start_time  # Calculate the total runtime
    
    result['total_runtime'] = total_runtime  # Add the total runtime to the result dictionary

    if as_list:
        return [(key, value['total_tokens']) for key, value in result.items()]
    else:
        return result
    
def apply_callback_to_column(data, callback, target_column, output_column):
    modified_data = {'json': [], 'csv': []}
    
    for file_type, dfs in data.items():
        for df in dfs:
            # Check if the target column exists in the dataframe
            if target_column in df.columns:
                # Apply the callback function to the target column
                df[output_column] = df[target_column].apply(callback)
            
            # Store the modified dataframe
            modified_data[file_type].append(df)
    
    return modified_data


# In[19]:


token_estimated_data = apply_callback_to_column(data, num_tokens_from_string, 'code', 'token_count')
token_estimated_data['csv'] = pd.concat(token_estimated_data['csv'], axis=0, ignore_index=True, sort=False)
token_estimated_data['json'] = pd.concat(token_estimated_data['json'], axis=0, ignore_index=True, sort=False)

# #### And a Cost Estimator

# In[20]:


def input_cost_calculator(num_tokens: int, model_name: Optional[str] = None, context: Optional[str] = None) -> Union[Dict[str, Dict[str, float]], float]:
    pricing_per_1K = {
        "GPT-4": {"8K_context": 0.03, "32K_context": 0.06},
        "GPT-3.5_Turbo": {"4K_context": 0.0015, "16K_context": 0.003},
        "FineTuningModels": {
            "babbage-002": {"training": 0.0004, "input_usage": 0.0016, "output_usage": 0.0016},
            "davinci-002": {"training": 0.0060, "input_usage": 0.0120, "output_usage": 0.0120},
            "GPT-3.5_Turbo": {"training": 0.0080, "input_usage": 0.0120, "output_usage": 0.0160}
        },
        "EmbeddingModels": {"Ada_v2": 0.0001},
        "BaseModels": {"babbage-002": 0.0004, "davinci-002": 0.0020}
    }

    if model_name and context:
        if model_name in pricing_per_1K:
            if context in pricing_per_1K[model_name]:
                price_per_1K = pricing_per_1K[model_name][context]
                return price_per_1K * (num_tokens / 1000)
        return "Invalid model_name or context"

    cost_calculations = {}
    for model, contexts in pricing_per_1K.items():
        cost_calculations[model] = {}
        if isinstance(contexts, dict):
            for context, price_per_1K in contexts.items():
                if isinstance(price_per_1K, dict):
                    cost_calculations[model][context] = {}
                    for sub_context, sub_price in price_per_1K.items():
                        cost_calculations[model][context][sub_context] = sub_price * (num_tokens / 1000)
                else:
                    cost_calculations[model][context] = price_per_1K * (num_tokens / 1000)
        else:
            cost_calculations[model] = contexts * (num_tokens / 1000)
    return cost_calculations

token_estimated_data['json']['GPT-4 (8K)'] = token_estimated_data['json']['token_count'].apply(lambda x: input_cost_calculator(x, model_name="GPT-4", context="8K_context"))
token_estimated_data['json']['GPT-4 (32K)'] = token_estimated_data['csv']['token_count'].apply(lambda x: input_cost_calculator(x, model_name="GPT-4", context="32K_context"))

token_estimated_data['csv']['GPT-4 (8K)'] = token_estimated_data['csv']['token_count'].apply(lambda x: input_cost_calculator(x, model_name="GPT-4", context="8K_context"))
token_estimated_data['csv']['GPT-4 (32K)'] = token_estimated_data['csv']['token_count'].apply(lambda x: input_cost_calculator(x, model_name="GPT-4", context="32K_context"))


token_estimated_data['json']['estimates'] = token_estimated_data['json']['token_count'].apply(input_cost_calculator)
token_estimated_data['csv']['estimates'] = token_estimated_data['csv']['token_count'].apply(input_cost_calculator)

# Create a boolean mask to filter rows containing '/comparison' in 'filename' column
mask = token_estimated_data['csv']['filename'].str.contains('/comparison')

# Use the mask to filter the DataFrame
filtered_df = token_estimated_data['csv'][mask]


class TokenManager:
    def __init__(self, max_tokens_per_minute=10000):
        self.max_tokens_per_minute = max_tokens_per_minute
        self.tokens_used_in_last_minute = 0
        self.last_request_time = time.time()

    def tokens_available(self):
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request >= 60:
            self.tokens_used_in_last_minute = 0
            self.last_request_time = current_time

        return self.max_tokens_per_minute - self.tokens_used_in_last_minute

    def update_tokens_used(self, tokens_used):
        if not isinstance(tokens_used, int) or tokens_used < 0:
            raise ValueError("Invalid token count")
        self.tokens_used_in_last_minute += tokens_used

    def get_usage_stats(self):
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time

        if time_since_last_request >= 60:
            self.tokens_used_in_last_minute = 0
            self.last_request_time = current_time

        absolute_usage = self.tokens_used_in_last_minute
        percent_usage = (self.tokens_used_in_last_minute / self.max_tokens_per_minute) * 100

        return absolute_usage, percent_usage

token_manager = TokenManager()

def fetch_response(code_snippet, system_prompt):
    """
    Fetches a response using the OpenAI API.

    Parameters:
        code_snippet (str): The code snippet to be processed.
        system_prompt (str): The system prompt.

    Returns:
        dict: The response from the API.
    """
    while token_manager.tokens_available() < 2500:  # Assume a minimum of 10 tokens per request
        time.sleep(1)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": code_snippet}
            ]
        )
        tokens_used = response['usage']['total_tokens']
        token_manager.update_tokens_used(tokens_used)
        return response
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def main(df, system_prompt):
    """
    Processes a DataFrame of code snippets.

    Parameters:
        df (pd.DataFrame): The DataFrame containing code snippets.
        system_prompt (str): The system prompt.

    Returns:
        pd.DataFrame: The DataFrame with the added output column.
    """
    responses = [fetch_response(row['code'], system_prompt) for _, row in df.iterrows()]
    df['output'] = responses
    return df

# Run the main function
new_df = asyncio.run(main(filtered_df, system_prompt_1))

# Optionally, save the new DataFrame to a CSV file
new_df.to_csv('comparison_openAI_output_LXIX.csv', index=False)




