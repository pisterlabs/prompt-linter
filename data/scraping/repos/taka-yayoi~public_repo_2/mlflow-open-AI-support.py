# Databricks notebook source
# MAGIC %md
# MAGIC # MLflowのOpenAI APIサポート
# MAGIC
# MAGIC - [MLflow 2\.3のご紹介：ネイティブLLMのサポートと新機能による強化 \- Qiita](https://qiita.com/taka_yayoi/items/431fa69430c5c6a5e741)
# MAGIC - [DatabricksでMLflow 2\.3のOpenAI APIのサポートを試す \- Qiita](https://qiita.com/taka_yayoi/items/a058484e6c0abbfbc476)
# MAGIC
# MAGIC **前提:** DBR13.0ML

# COMMAND ----------

# MAGIC %md
# MAGIC ## ライブラリのインストール

# COMMAND ----------

# MAGIC %pip install tiktoken
# MAGIC %pip install openai
# MAGIC %pip install mlflow==2.3.1

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## モデルの記録
# MAGIC
# MAGIC [Databricksにおけるシークレットの管理 \- Qiita](https://qiita.com/taka_yayoi/items/338ef0c5394fe4eb87c0)

# COMMAND ----------

import os
import mlflow
import openai

# 事前にDatabricksシークレットにキーを openai_api_key としてOpen AI APIキーを登録しておきます
# 以下の環境変数にはシークレットのスコープを指定します
os.environ["MLFLOW_OPENAI_SECRET_SCOPE"] = "demo-token-takaaki.yayoi"

# COMMAND ----------

with mlflow.start_run():
    model_info = mlflow.openai.log_model(
        model="gpt-3.5-turbo",
        task=openai.ChatCompletion,
        messages=[{"role": "system", "content": "あなたはDatabricksの専門家です"}],
        artifact_path="model"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC 確認用JSON
# MAGIC
# MAGIC ```
# MAGIC {"inputs": ["Databricksとは"]}
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ## サービングエンドポイント

# COMMAND ----------

import streamlit as st 
import numpy as np 
from PIL import Image
import base64
import io

import os
import requests
import numpy as np
import pandas as pd

import json

st.header('OpenAI Chatbot on Databrikcs')
st.write('''
- [MLflow 2\.3のご紹介：ネイティブLLMのサポートと新機能による強化 \- Qiita](https://qiita.com/taka_yayoi/items/431fa69430c5c6a5e741)
- [DatabricksでMLflow 2\.3のOpenAI APIのサポートを試す \- Qiita](https://qiita.com/taka_yayoi/items/a058484e6c0abbfbc476)
''')

def score_model(prompt):
  # 1. パーソナルアクセストークンを設定してください
  # 今回はデモのため平文で記載していますが、実際に使用する際には環境変数経由で取得する様にしてください。
  token = "<パーソナルアクセストークン>"
  #token = os.environ.get("DATABRICKS_TOKEN")

  # 2. サービングエンドポイントのURLを設定してください
  url = '<サービングエンドポイントURL>'
  headers = {'Authorization': f'Bearer {token}'}
  #st.write(token)
  
  data_json_str = f"""
  {{"inputs":
["{prompt}"]
}}
"""
  #st.write(data_json_str)

  data_json = json.loads(data_json_str)
   
  response = requests.request(method='POST', headers=headers, url=url, json=data_json)
  #st.write(response)

  if response.status_code != 200:
    raise Exception(f'Request failed with status {response.status_code}, {response.text}')
  return response.json()

prompt = st.text_input("プロンプト")

if prompt != "":
  response = score_model(prompt)
  st.write(response['predictions'][0])

# COMMAND ----------

# MAGIC %md
# MAGIC 上のPythonコードをchatbot.pyというファイルに保存し、以下を実行します。
# MAGIC
# MAGIC ```
# MAGIC streamlit run chatbot.py
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC # END
