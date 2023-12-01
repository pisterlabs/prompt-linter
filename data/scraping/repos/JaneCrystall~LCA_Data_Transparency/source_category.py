import json
import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from retrying import retry

load_dotenv()
openai_client = OpenAI()


@retry(wait_fixed=300, stop_max_attempt_number=5)
def query_func_calling(query: str):
    # Step 1: send the conversation and available functions to the model
    messages = [
        {
            "role": "user",
            "content": query,
        }
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_source_category",
                "description": "Generating category for sources of LCA database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "source_category": {
                            "title": "source_category",
                            "description": "Generate category for sources of LCA database. The category must be one of the following: 'Article in periodical', 'Chapter in anthology', 'Monograph', 'Industry report', 'Standard', 'Environmental Product Declaration (EPD)','Patent', 'Statistical documents', 'Software or database', 'Personal communication','Direct measurement'.",
                            "type": "string",
                        },
                    },
                    "required": [
                        "source_category",
                    ],
                },
            },
        }
    ]
    response = openai_client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=messages,
        tools=tools,
        temperature=0.0,
        tool_choice="auto",  # auto is default, but we'll be explicit
    )
    response_message = response.choices[0].message
    response_string = response_message.tool_calls[0].function.arguments
    return response_string

file_path = 'Gabi_source.xlsx'

# 读取Excel文件
df = pd.read_excel(file_path)
source_info = df['source_info']
for source in source_info:
    source_category = query_func_calling(source)
    df.loc[df['source_info'] == source, 'source_category'] = source_category

df.to_excel('Gabi_source_category.xlsx', index=False)