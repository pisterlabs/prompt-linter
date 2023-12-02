import json
import os

import oss2
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from xata import XataClient

from src.models.lca_models import Source

load_dotenv()

oss_access_key_id = os.getenv("OSS_ACCESS_KEY_ID")
oss_access_key_secret = os.getenv("OSS_ACCESS_KEY_SECRET")
oss_endpoint = os.getenv("OSS_ENDPOINT")
oss_bucket = os.getenv("OSS_BUCKET")

oss_auth = oss2.Auth(oss_access_key_id, oss_access_key_secret)
oss_bucket = oss2.Bucket(oss_auth, oss_endpoint, oss_bucket)

xata_api_key = os.getenv("XATA_API_KEY")
xata_database_url = os.getenv("XATA_DATABASE_URL")

xata_client = XataClient(api_key=xata_api_key, db_url=xata_database_url)
openai_client = OpenAI()


async def query_lca_source(query: str):
    results = xata_client.data().search_table("source", {"query": query})

    lca_source_results = []

    for record in results["records"]:
        for field in ["classification", "data_set_format", "short_name"]:
            if field in record and isinstance(record[field], str):
                record[field] = json.loads(record[field])
        lca_source_result = Source(**record)
        lca_source_results.append(lca_source_result)

    return lca_source_results


async def process_query_func_calling(query: str):
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
                "name": "get_process_name",
                "description": "Generating keywords for LCA process database fulltext searching",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "phraseChinese": {
                            "title": "keywords_in_Chinese",
                            "description": "Extract Chinese keywords including location, time and technique information. Translate the English phrase into accurate Chinese if it is not already in Chinese.",
                            "type": "string",
                        },
                        "phraseEnglish": {
                            "title": "keywords_in_English",
                            "description": "Extract English keywords including location, time and technique information. Translate the Chinese phrase into accurate English if it is not already in English.",
                            "type": "string",
                        },
                        "alias": {
                            "title": "alias of process keywords in English and Chinese",
                            "description": "Only provide alias that are commonly agreed upon or widely recognized in both English and Chinese.",
                            "type": "array",
                            "items": {
                                "type": "string",
                            },
                        },
                        "synonym": {
                            "title": "synonym of process keywords in English and Chinese",
                            "description": "Only provide synonym that are commonly agreed upon or widely recognized in both English and Chinese.",
                            "type": "array",
                            "items": {
                                "type": "string",
                            },
                        },
                        "abbreviation": {
                            "title": "abbreviation of process keywords",
                            "description": "Only provide abbreviations that are commonly agreed upon or widely recognized in both English and Chinese.",
                            "type": "array",
                            "items": {
                                "type": "string",
                            },
                        },
                    },
                    "required": [
                        "phraseChinese",
                        "phraseEnglish",
                        "alias",
                        "synonym",
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
    response_json = json.loads(response_string)

    # 创建一个空列表用于存放所有的字符串
    response_list = []

    # 遍历字典中的每个值
    for value in response_json.values():
        # 如果值是字符串，直接添加到列表中
        if isinstance(value, str):
            response_list.append(value)
        # 如果值是列表，将列表中的每个元素（字符串）添加到列表中
        elif isinstance(value, list):
            response_list.extend(value)

    return response_list


async def process_search(lca_process_query: str):
    query_list = await process_query_func_calling(lca_process_query)

    unique_results = {}
    for query in query_list:
        fuzzy_search_response = xata_client.data().search_table(
            "process",
            {
                "query": query,
                "target": ["name", "description"],
                "fuzziness": 0,
            },
        )
        fuzzy_search_results = fuzzy_search_response["records"][:10]

        # 遍历查询结果
        for result in fuzzy_search_results:
            uuid = result["uuid"]
            score = result["xata"].get("score")
            # 如果uuid不在字典中，则添加该记录
            if uuid not in unique_results:
                score = result.pop("xata", {}).get("score", None)
                result["score"] = score
                unique_results[uuid] = result

    # 创建一个 DataFrame
    df = pd.DataFrame.from_dict(unique_results, orient="index")

    # 按照 'score' 排序，从高到低
    df_sorted = df.sort_values(by='score', ascending=False)
    df_subset = df_sorted[['uuid','name', 'description']]

    # 选择前 5条记录
    df_top = df_subset.head(5)

    df_top_json = df_top.to_json(orient='records')

    return df_top_json
