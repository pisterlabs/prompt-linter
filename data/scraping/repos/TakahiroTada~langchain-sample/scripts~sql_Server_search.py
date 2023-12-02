import aiohttp
import os
from dataclasses import dataclass, field, asdict
from dataclasses_json import dataclass_json
from enum import Enum
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
import logging
from sqlalchemy.engine import URL
import openai
import re
import json

load_dotenv()

# SQL ServerのURLを構築する
connection_url = URL.create(
    "mssql+pyodbc",
    username=os.getenv("SQL_SERVER_USER_NAME"),
    password=os.getenv("SQL_SERVER_PASSWORD"),
    host=os.getenv("SQL_SERVER_HOST"),
    port=int(os.getenv("SQL_SERVER_PORT")),
    database=os.getenv("SQL_SERVER_DATABASE"),
    query={"driver": os.getenv("SQL_SERVER_DRIVER_NAME")})

# SQL Serverの接続をセットアップする
db = SQLDatabase.from_uri(database_uri=connection_url, include_tables=os.getenv("SQL_SERVER_INCLUDE_TABLES").split(','))

# LLMモデルをセットアップする
openai.api_type = os.getenv("OPENAI_API_TYPE")   
openai.api_base = os.getenv("OPENAI_API_BASE")  
openai.api_version = os.getenv("OPENAI_API_VERSION")  
openai.api_key = os.getenv("OPENAI_API_KEY")  
llm = ChatOpenAI(
    model_kwargs={"engine" : os.getenv("OPENAI_LLM_DEPLOYMENT_NAME")},  
    temperature=int(os.getenv("OPENAI_LLM_TEMPERATURE")),   
    openai_api_key = openai.api_key
)

# LLMモデルにSQL Serverの接続を渡す
db_chain = SQLDatabaseChain.from_llm(llm=llm, db=db, verbose=True)

# LLMモデルに質問を投げる
ret = db_chain.run("ユーザーテーブルのなかで、最近登録したユーザーを10人抽出して、ユーザーのIDと名前とレベルを教えてください。")

# LLMモデルの回答を表示する
logging.info("before value:%s", ret)

retjson = None
if('Question:' in ret and 'SQLQuery:' in ret):
    ret = re.split('Question:|SQLQuery:', ret)
    retjson = {
        'value':  ret[0].strip(),
        'Question' : ret[1].strip(),
        'SQLQuery' : ret[2].strip()
    }
elif('Question:' in ret):
    ret = re.split('Question:', ret)
    retjson = {
        'value':  ret[0].strip(),
        'Question' : ret[1].strip()
    }
elif('SQLQuery:' in ret):
    ret = re.split('SQLQuery:', ret)
    retjson = {
        'value':  ret[0].strip(),
        'SQLQuery' : ret[1].strip()
    }
elif('value:' in ret):
    ret = re.split('value:', ret)
    retjson = {
        'value':  ret[0].strip()
    }
else:
    retjson = {
        'value':  ret.strip()
    }                   
logging.info("after value(ret):%s", json.dumps(retjson, ensure_ascii=False))
logging.info("after value(json):%s", json.dumps(retjson, ensure_ascii=False))

