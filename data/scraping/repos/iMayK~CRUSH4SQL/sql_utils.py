import os
import time
import json
from collections import defaultdict
import pandas as pd

import openai

from .prompts import PROMPTS

file_dir = os.path.dirname(os.path.abspath(__file__))
json_file_path = os.path.join(file_dir, 'relation_map_for_unclean.json')
RELATION_MAP = json.load(open(json_file_path))

def create_schema(selected_lst):
    file_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(file_dir, 'ndap_super.json')
    ndap_meta = json.load(open(file_path))
    schema = defaultdict(list)
    for item in selected_lst:
        table_source = RELATION_MAP[item]['source']
        table_code = RELATION_MAP[item]['code']
        table_name = RELATION_MAP[item]['table']
        prefix = f'{table_source} {table_name} {table_code}'
        column_name = item[len(prefix):].strip()
        full_table_name = f'{table_name} {table_code}'.strip()
        schema[full_table_name].append(column_name)

    for key in schema.keys():
        table_code = key.split(' ')[-1]
        key_columns = ndap_meta[table_code]['key_columns']
        schema[key] = schema[key] + [key_column for key_column in key_columns if key_column.strip() not in schema[key]]

    return schema

def create_schema_str(schema):
    schema_str = ''
    for tbl, col_info in schema.items():
        schema_str += f'"{tbl}" (\n'
        for col in col_info:
            schema_str += f'\t"{col},"\n'
        schema_str += '),\n'
    return schema_str

def generate(prompt, api_type, api_key, endpoint, api_version):
    if api_type == 'azure':
        openai.api_type = "azure"
        openai.api_key = api_key
        openai.api_base = endpoint
        openai.api_version = api_version 

        deployment_name = 'chatgpt-35-16k'
        response = '' 
        try:
            response = openai.ChatCompletion.create(
                engine = deployment_name,
                messages=[
                        {"role": "system", "content": prompt},
                    ],
                temperature=0
            )
        except Exception as e:
            time.sleep(10)
            try:
                response = openai.ChatCompletion.create(
                    engine = deployment_name,
                    messages=[
                            {"role": "system", "content": prompt},
                        ],
                    temperature=0
                )
            except Exception as e:
                return f"An error occured: {str(e)}"
        return response['choices'][0]['message']['content'] 
    else:
        openai.api_key = api_key
        model_name = 'gpt-3.5-turbo-16k'
        response = ''
        try:
            response = openai.ChatCompletion.create(
                model=model_name,
                messages=[
                        {"role": "system", "content": prompt},
                    ],
                temperature=0
            )
        except Exception as e:
            time.sleep(10)
            try:
                response = openai.ChatCompletion.create(
                    model=model_name,
                    messages=[
                            {"role": "system", "content": prompt},
                        ],
                    temperature=0
                )
            except Exception as e:
                return f"An error occured: {str(e)}"
        return response.choices[0].message.content

def generate_sql(item, api_type, api_key, endpoint, api_version, prompting_type='base', fewshot_examples=[]):
    schema = create_schema(item['docs'])
    schema_str = create_schema_str(schema)

    question = item['question']

    prompt_template = PROMPTS[prompting_type]

    if prompting_type == 'base':
        prompt = prompt_template.format(question, schema_str, question)
    else:
        examples = ""
        for idx, item in enumerate(fewshot_examples):
            examples += f"Question {idx+1}: {item['question']}\nSQL: {item['sql']}\n\n"
        prompt = prompt_template.format(question, schema_str, examples, question)  
    pred_sql = generate(prompt, api_type, api_key, endpoint, api_version)

    return prompt, pred_sql, schema_str




