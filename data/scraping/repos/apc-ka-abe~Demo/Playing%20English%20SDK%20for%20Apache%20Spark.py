# Databricks notebook source
# MAGIC %md
# MAGIC ## English SDK for Apache Sparkの説明

# COMMAND ----------

# MAGIC %md
# MAGIC # 自然言語でSpark DataFrameを操作してみましょう！

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC <div  style="text-align: center; line-height: 0; padding-top: 9px;">
# MAGIC   <img src="https://github.com/apc-ka-abe/Demo/blob/f946ce79cb97dd775095dd23d76a28842b6028dc/English_SDK.png" width="60%">
# MAGIC </div>

# COMMAND ----------

# MAGIC %md
# MAGIC ## セットアップ

# COMMAND ----------

# MAGIC %pip install pyspark-ai

# COMMAND ----------

# MAGIC %md
# MAGIC ### Azure OpenAIのtokenを取得

# COMMAND ----------

import os
import openai
 
# `Azure`固定
openai.api_type = "azure"
 
# Azure Open AI のエンドポイント
openai.api_base = "https://ka-abe-azureopen-api-japan-east.openai.azure.com/"
 
# Azure Docs 記載の項目
openai.api_version = "2023-05-15"
 
# Azure Open AI のキー
os.environ["OPENAI_API_KEY"] = dbutils.secrets.get("demo-token-ka.abe", "azure_openai_api_key")
 
# デプロイ名
# deployment_id = "ka-abe-gpt-turbo"
deployment_id = "ka-abe-gpt-4"

# デプロイしたモデル名
# model_name = "gpt-35-turbo"
model_name = "gpt-4"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Chainを作成

# COMMAND ----------

from langchain.chat_models import ChatOpenAI
from pyspark_ai import SparkAI
 
llm = ChatOpenAI(
    deployment_id=deployment_id,
    model_name=model_name,
    temperature=0, 
)
 
spark_ai = SparkAI(llm=llm, verbose=True)
spark_ai.activate()

# COMMAND ----------

# MAGIC %md
# MAGIC ## URLからデータフレームを生成

# COMMAND ----------

# MAGIC %md
# MAGIC Web上の表データがあるURLを指定することで、DataFrameを作成できます

# COMMAND ----------

# 例：日本の歴代総理大臣の一覧をwikipediaから読み込み、DataFrameを作成する
auto_df = spark_ai.create_df("https://en.wikipedia.org/wiki/List_of_prime_ministers_of_Japan", ['Prime ministers Office(lifespan)', 'Term of office', 'Mandate'])
display(auto_df)

# COMMAND ----------

# アメリカの歴代大統領
auto_df_2 = spark_ai.create_df("https://en.wikipedia.org/wiki/List_of_presidents_of_the_United_States", ["president", "vice_president"])
display(auto_df_2)

# COMMAND ----------

# Let's play
# https://memorva.jp/ranking/unfpa/who_whs_population.php
tmp_df = spark_ai.create_df("<input>")
display(tmp_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ##　DataFrameを読み込む

# COMMAND ----------

# MAGIC %md
# MAGIC 使用するデータ：ニューヨークにおけるタクシーの乗降記録データ(Databricksのサンプルデータセット)

# COMMAND ----------

catalog_name = 'ka_abe'
schema_name = 'sample'
table_name = 'nyctaxi'
taxi_df = spark.read.table(f'{catalog_name}.{schema_name}.{table_name}')

# COMMAND ----------

display(taxi_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## DataFrameの説明

# COMMAND ----------

taxi_df.ai.explain()

# COMMAND ----------

# MAGIC %md
# MAGIC ## DataFrameを検証(判定)

# COMMAND ----------

# MAGIC %md
# MAGIC DataFrameに異常値が含まれていないか等を検証する

# COMMAND ----------

# 欠損値がないか判定する
taxi_df.ai.verify("expect no NULL values")

# COMMAND ----------

# ０時から24時までの時間内か検証する
taxi_df.ai.verify("The tpep_dropoff_datetime column is in the range 00:00:00-23:59:59")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transformation

# COMMAND ----------

# MAGIC %md
# MAGIC データの変換方法を指示し、必要なDataFrameをすぐに用意できる

# COMMAND ----------

# GPT-4
transformed_taxi_df = spark_ai.transform_df(taxi_df, "get all data")
transformed_taxi_df.display()

# COMMAND ----------

# GPT-4
transformed_taxi_df_jp = spark_ai.transform_df(taxi_df, "すべてのデータを取得する")
transformed_taxi_df_jp.display()　

# COMMAND ----------

# tpep_pickup_datetimeカラムとtpep_dropoff_datetimeカラム間の時間差(分)を表すride_timeというカラムを作成します。
transformed_taxi_df = spark_ai.transform_df(taxi_df, "Create a column named ride_time for the time difference (minutes) between the tpep_pickup_datetime and tpep_dropoff_datetime columns")
transformed_taxi_df.display()

# COMMAND ----------

transformed_taxi_df_jp = spark_ai.transform_df(taxi_df, "tpep_pickup_datetimeカラムとtpep_dropoff_datetimeカラム間の時間差(分)を表すride_timeというカラムを作成します。")
transformed_taxi_df_jp.display()

# COMMAND ----------

# Let's Playing
transformed_taxi_df_jp = spark_ai.transform_df(taxi_df, "<input>")
transformed_taxi_df_jp.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ## グラフの作成

# COMMAND ----------

# MAGIC %md
# MAGIC 自然言語で図の作成も簡単に行うことが可能です。

# COMMAND ----------

# 2019-12-26 00:00:00から30分後までのfare_amountを散布図にプロットします。
taxi_df.ai.plot("Plot the fare_amount from 2019-12-26 00:00:00 to 30 minutes later in a scatter plot.")

# COMMAND ----------

# Pythonで書く場合。。。かなり長い
from pyspark.sql import SparkSession
import pandas as pd
import plotly.express as px

# Start Spark session
spark = SparkSession.builder.appName('nyctaxi').getOrCreate()

# Load the data into a DataFrame
df = spark.sql("SELECT * FROM ka_abe.sample.nyctaxi")

# Convert Spark DataFrame to Pandas DataFrame
df_pd = df.toPandas()

# Convert the 'tpep_pickup_datetime' column to datetime
df_pd['tpep_pickup_datetime'] = pd.to_datetime(df_pd['tpep_pickup_datetime'])

# Filter the data for the desired date range
start_date = '2019-12-26 00:00:00'
end_date = '2019-12-26 00:30:00'
mask = (df_pd['tpep_pickup_datetime'] > start_date) & (df_pd['tpep_pickup_datetime'] <= end_date)
df_pd = df_pd.loc[mask]

# Plot the data using plotly
fig = px.scatter(df_pd, x='tpep_pickup_datetime', y='fare_amount', title='Fare Amount from 2019-12-26 00:00:00 to 30 minutes later')
fig.show()

# COMMAND ----------

# 2019-12-26 00:00:00から30分までのtotal_amountの箱ひげ図を作成する
taxi_df.ai.plot("Create a box-and-whisker diagram of total_amount from 2019-12-26 00:00:00 to 30 minutes")

# COMMAND ----------

# Let's play
taxi_df.ai.plot("")

# COMMAND ----------

# MAGIC %md
# MAGIC ## UDF（ユーザー定義関数）の作成

# COMMAND ----------

# MAGIC %md
# MAGIC UDF(user-defined-function:UDF)とは、ユーザーが定義したカスタムロジックの処理を実行する関数

# COMMAND ----------

# MAGIC %md
# MAGIC チップの金額に対してランクづけをします。</br>
# MAGIC ランクの閾値は自動で決めてもらいます。

# COMMAND ----------

@spark_ai.udf
def Rank_tip(tip: str) -> str:
    """Rank for tip_amount"""
    return tip

# COMMAND ----------

from pyspark.sql import SparkSession

spark = SparkSession.builder.getOrCreate()
spark.udf.register("Rank_tip", Rank_tip)
# percentGrades = [(1, 97.8), (2, 72.3), (3, 81.2)]
# df = spark.createDataFrame(percentGrades, ["student_id", "grade_percent"])
taxi_df.selectExpr("tip_amount", "Rank_tip(tip_amount)").show()
