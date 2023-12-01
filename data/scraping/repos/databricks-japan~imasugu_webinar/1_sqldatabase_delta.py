# Databricks notebook source
# MAGIC %md # SQLDatabaseChainを使った自然言語でのデータ問い合わせ
# MAGIC
# MAGIC 今回はLangchainの SQLDatabaseChainというChainを利用して、Databricks内のデータに問い合わせをしてみます。<br>
# MAGIC LLMとしては、OpenAIのモデルを利用しております。（他にAzure OpenAIにも対応）
# MAGIC
# MAGIC <img src='https://sajpstorage.blob.core.windows.net/maruyama/webinar/llm/llm-database-1.png' width='800' />
# MAGIC
# MAGIC Databricks Runtime 13.1 ML 以降をご利用ください。

# COMMAND ----------

!pip install databricks-sql-connector openai langchain==0.0.205 
dbutils.library.restartPython()

# COMMAND ----------

import os
import openai
from langchain import SQLDatabase, SQLDatabaseChain
from langchain.chat_models import ChatOpenAI

# COMMAND ----------

# MAGIC %md ## OpenAI APIの Secret Keyを取得し指定します。
# MAGIC
# MAGIC OpenAI API Secret Keyは、[こちら](https://platform.openai.com/account/api-keys)から取得してください。<br>
# MAGIC 以下はDatabricksの[Secret機能](https://qiita.com/maroon-db/items/6e2d86919a827bd61a9b)を利用しております。ScopeとKeyを変更ください。<br>
# MAGIC

# COMMAND ----------

OPENAI_API_KEY = dbutils.secrets.get("<scope>", "<key>") 

# COMMAND ----------

# MAGIC %md ## SQLDatabaseChainの設定
# MAGIC
# MAGIC 今回対象とするカタログやスキーマ、テーブルなどを指定します。<br><br>
# MAGIC SQLDatabaseChainの使い方については、こちらをご覧ください。<br>
# MAGIC https://python.langchain.com/docs/ecosystem/integrations/databricks
# MAGIC
# MAGIC Qiita: [(翻訳) LangChainのSQLDatabaseChain](https://qiita.com/taka_yayoi/items/7759f31341b91bc707f4)

# COMMAND ----------

#######################################################
## 今回検索対象とするカタログとスキーマを選択します。   #######
#######################################################

catalog = "samples"
schema = "nyctaxi" 
tables  = ["trips"] 

# LLM. 今回は ChatGPT3.5を利用
llm = ChatOpenAI(model_name='gpt-3.5-turbo',temperature=0, verbose=True, openai_api_key=OPENAI_API_KEY)

## Langchain SQLDatabaseChain
db = SQLDatabase.from_databricks(catalog=catalog, schema=schema,include_tables=tables)

db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# COMMAND ----------

# MAGIC %md ### 動作確認
# MAGIC
# MAGIC

# COMMAND ----------

# DBTITLE 1,その前に対象データを確認
display(spark.table(f"{catalog}.{schema}.trips"))

# COMMAND ----------

db_chain.run("18時以降の平均の運賃は?")

# COMMAND ----------

db_chain.run("乗車時間の平均時間は？")

# COMMAND ----------

# MAGIC %md ## 検索履歴を保存するようにカスタマイズ
# MAGIC
# MAGIC ユーザーがどのような問い合わせをしたのかを履歴としてDelta Tableに保存するように、カスタマイズします。<br>
# MAGIC 保存するカタログやスキーマ、テーブル名などを変更してからご利用ください。

# COMMAND ----------

# DBTITLE 1,delta_chain 関数の作成
from pyspark.sql.types import StructType, StructField, StringType, TimestampType
from pyspark.sql.functions import current_timestamp, date_format

# 保存先テーブルの作成
d_catalog = "<catalog>"
d_schema = "llmdb"
d_table = "query_history"

spark.sql(f"create catalog if not exists {d_catalog}")
spark.sql(f"create schema if not exists {d_catalog}.{d_schema}")

def delta_chain(input):

  # スキーマを定義
  schema = StructType([
      StructField('query', StringType(), True),
      StructField('result', StringType(), True)
  ])
  
  # SQLDatabaseChainにて問い合わせ
  result = db_chain(input)

  # DataFrameを作成＆保存
  df = spark.createDataFrame( [tuple(result.values())], schema=schema)
  df = df.withColumn('time',current_timestamp()).select('time','query','result')
  df.write.mode("append").saveAsTable(f"{d_catalog}.{d_schema}.{d_table}")
  return result['result']


# COMMAND ----------

# DBTITLE 1,動作テスト
delta_chain("平均の運賃は？")

# COMMAND ----------

# DBTITLE 1,history tableの確認
display(spark.table(f"{d_catalog}.{d_schema}.{d_table}"))

# COMMAND ----------


