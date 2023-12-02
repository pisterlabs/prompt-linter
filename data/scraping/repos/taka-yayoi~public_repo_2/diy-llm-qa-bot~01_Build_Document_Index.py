# Databricks notebook source
# MAGIC %md このノートブックの目的は、QA Botアクセラレータを構成するノートブックを制御するさまざまな設定値を設定することです。このノートブックは https://github.com/databricks-industry-solutions/diy-llm-qa-bot から利用できます。

# COMMAND ----------

# MAGIC %md ## イントロダクション
# MAGIC
# MAGIC QA botアプリケーションが適切な回答で反応するように、質問に適したドキュメントのコンテンツをモデルに提供する必要があります。botがレスポンスを導き出すためにこれらのドキュメントの情報を活用するというアイデアです。
# MAGIC
# MAGIC 我々のアプリケーションでは、[Databricksドキュメント](https://docs.databricks.com/)、[Sparkドキュメント](https://spark.apache.org/docs/latest/)、[Databricksナレッジベース](https://kb.databricks.com/)から一連のドキュメントを抽出しました。Databricksナレッジベースは、FAQに対応し、高品質で詳細なレスポンスがあるオンラインフォーラムです。コンテキストを提供するためにこれらの3つのドキュメントソースを用いることで、我々のbotは深い専門性を持って、この領域において適切な質問に反応することができます。
# MAGIC
# MAGIC </p>
# MAGIC
# MAGIC <img src='https://brysmiwasb.blob.core.windows.net/demos/images/bot_data_processing4.png' width=700>
# MAGIC
# MAGIC </p>
# MAGIC
# MAGIC このノートブックでは、これらのドキュメントをロードし、別のプロセスを通じて一連のJSONドキュメントとして抽出し、Databricks環境におけるテーブルに格納します。ドキュメントに関するメタデータを収集し、高速なドキュメント検索や取得を可能にするインデックスを構成するベクトルストアに取り込みます。

# COMMAND ----------

# DBTITLE 1,必要ライブラリのインストール
# MAGIC %pip install langchain==0.0.166 tiktoken==0.4.0 openai==0.27.6 faiss-cpu==1.7.4

# COMMAND ----------

# DBTITLE 1,必要な関数のインポート
import pyspark.sql.functions as fn

import json

from langchain.text_splitter import TokenTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS

# COMMAND ----------

# DBTITLE 1,設定の取得
# MAGIC %run "./util/notebook-config"

# COMMAND ----------

# MAGIC %md ##Step 1: 生データをテーブルにロード
# MAGIC
# MAGIC ３つのドキュメントソースのスナップショットは、公開されているクラウドストレージからアクセスできます。最初のステップは、抽出されたドキュメントにアクセスすることです。`multiLine`オプションを用いて[JSON](https://spark.apache.org/docs/3.1.2/api/python/reference/api/pyspark.sql.DataFrameReader.json.html)を読み込むように設定されたSpark DataReaderを用いてこれらをテーブルにロードすることができます。

# COMMAND ----------

# DBTITLE 1,JSONデータをデータフレームに読み込み
raw = (
  spark
    .read
    .option("multiLine", "true")
    .json(
      f"{config['kb_documents_path']}/source/"
      )
  )

display(raw)

# COMMAND ----------

# MAGIC %md 
# MAGIC 以下のように、このデータをテーブルに永続化することができます：

# COMMAND ----------

# DBTITLE 1,データをテーブルに保存
# データをテーブルに保存
_ = (
  raw
    .write
    .format('delta')
    .mode('overwrite')
    .option('overwriteSchema','true')
    .saveAsTable('sources')
  )

# テーブルの行数をカウント
print(spark.table('sources').count())

# COMMAND ----------

# MAGIC %md ##Step 2: インデックスのためのデータを準備
# MAGIC
# MAGIC 新たにロードしたテーブルでは数多くのフィールドを利用できますが、我々のアプリケーションに適しているフィールドは：
# MAGIC
# MAGIC * text - ユーザーの質問に適した情報を含む可能性のあるドキュメントのテキストやナレッジベースの回答
# MAGIC * source - オンラインドキュメントをポイントするURL

# COMMAND ----------

# DBTITLE 1,生の入力の取得
raw_inputs = (
  spark
    .table('sources')
    .selectExpr(
      'text',
      'source'
      )
  ) 

display(raw_inputs)

# COMMAND ----------

# MAGIC %md 
# MAGIC それぞれのドキュメントで利用できるコンテンツには違いがありますが、いくつかのドキュメントは非常に長いものです。こちらに我々のデータセットにおける大規模なドキュメントのサンプルをしめします：

# COMMAND ----------

# DBTITLE 1,長いテキストサンプルの取得
long_text = (
  raw_inputs
    .select('text') # textフィールドのみを取得
    .orderBy(fn.expr("len(text)"), ascending=False) # 長さでソート
    .limit(1) # トップ1を取得
     .collect()[0]['text'] # テキストを変数に格納
  )

# long_textを表示
print(long_text)

# COMMAND ----------

# MAGIC %md 
# MAGIC ドキュメントのインデックスへの変換プロセスには、固定サイズのエンべディングへの変換が含まれます。エンべディングは座標のような一連の数値であり、テキストユニットのコンテンツを要約します。大きなエンべディングは、ドキュメントの比較的詳細な情報をキャプチャすることができますが、送信されるドキュメントが大きいほどエンべディングはコンテンツを汎化します。これは、誰かに段落や章、本全体を固定数の次元に要約することお願いするようなものです。スコープが大きいほど、要約では詳細を削ぎ落とし、テキストのより高次のコンセプトにフォーカスしなくてはなりません。
# MAGIC
# MAGIC これに対する一般的な戦略は、エンべディングを生成する際にテキストをチャンク(塊)に分割するというものです。これらのチャンクは意味のある詳細情報を捕捉するように十分大きなものである必要がありますが、汎化を通じてキーとなる要素が洗い流されてしまうほど大きなものである必要はありません。適切なチャンクサイズを決定することはサイエンスというより芸術の領域ではありますが、ここではこのステップで何が起きているのかを説明するために十分小さいチャンクサイズを使用しています：

# COMMAND ----------

# DBTITLE 1,テキストをチャンクに分割
text_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=10)
for chunk in text_splitter.split_text(long_text):
  print(chunk, '\n')

# COMMAND ----------

# MAGIC %md 
# MAGIC チャンク間のオーバーラップを指定していることに注意してください。これによってキーとなるコンセプトを捕捉する単語の分離を避ける助けになります。
# MAGIC
# MAGIC このデモではオーバーラップのサイズを非常に小さく設定していますが、オーバーラップのサイズがチャンク間でオーバーラップする単語の正確な数にうまく変換されていないことに気づくかもしれません。これは、単語でコンテンツを直接分割しているのではなく、テキストを構成する単語から導かれるバイトペアのエンコーディングトークンで分割しているためです。バイトペアエンコーディングの詳細は[こちら](https://huggingface.co/learn/nlp-course/chapter6/5?fw=pt)で確認できますが、多くのLLMアルゴリズムでテキストを圧縮するためによく適用されるメカニズムであることに注意してください。

# COMMAND ----------

# MAGIC %md 
# MAGIC ドキュメント分割のコンセプトを理解したところで、我々のドキュメントをチャンクに分割する関数を記述し、データに適用しましょう。このステップでは、最終的に情報を送信するChat-GPTモデルで指定されている[制限](https://help.openai.com/en/articles/4936856-what-are-tokens-and-how-to-count-them)に合わせて、チャンクサイズとオーバーラップサイズを大きな値に設定していることに注意してください。これらの値をさらに大きくすることは可能ですが、それぞれのChat-GPTモデルでは現在は固定数の *トークン* が許可されており、ユーザープロンプト全体と生成されるレスポンスはこのトークンリミットに収まらなくてはいけないことに注意してください。さもないと、エラーが生成されます：

# COMMAND ----------

# DBTITLE 1,チャンクの設定
chunk_size = 3500
chunk_overlap = 400

# COMMAND ----------

# DBTITLE 1,入力をチャンクに分割
@fn.udf('array<string>')
def get_chunks(text):

  # トークン化ユーティリティのインスタンスを作成
  text_splitter = TokenTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
  
  # テキストをチャンクに分割
  return text_splitter.split_text(text)


# テキストをチャンクに分割
chunked_inputs = (
  raw_inputs
    .withColumn('chunks', get_chunks('text')) # テキストをチャンクに分割
    .drop('text')
    .withColumn('num_chunks', fn.expr("size(chunks)"))
    .withColumn('chunk', fn.expr("explode(chunks)"))
    .drop('chunks')
    .withColumnRenamed('chunk','text')
  )

  # 変換データを表示
display(chunked_inputs)

# COMMAND ----------

# MAGIC %md ##Step 3: ベクトルストアの作成
# MAGIC
# MAGIC データをチャンクに分割することで、これらのレコードを検索可能なエンべディングに変換することができます。最初のステップは、コンテンツから変換されるドキュメントに関するメタデータとコンテンツを分割することです：

# COMMAND ----------

# DBTITLE 1,入力を検索可能なテキストとメタデータに分割
# 入力をpandasデータフレームに変換
inputs = chunked_inputs.toPandas()

# 検索可能なテキスト要素を抽出
text_inputs = inputs['text'].to_list()

# メタデータの抽出
metadata_inputs = (
  inputs
    .drop(['text','num_chunks'], axis=1)
    .to_dict(orient='records')
  )

# COMMAND ----------

# MAGIC %md 
# MAGIC 次に、データをロードするベクトルストアを初期化します。ベクトルストアに馴染みがない方に説明しますが、これらはテキストをエンべディングとして格納することに特化し、コンテンツの類似度に基づいて高速な検索を可能とする特殊なデータベースです。ここでは、Facebook AI Researchによって開発された[FAISS vector store](https://faiss.ai/)を使用します。これは高速、軽量で、我々のシナリオにおいて理想的な特性を持っています。
# MAGIC
# MAGIC ベクトルストアの設定で鍵となるのは、ドキュメントとすべての検索可能なテキストの両方をエンべディング(ベクトル)に変換するために使用するエンべディングモデルを用いて設定するということです。どのエンべディングを採用するのかを検討する際、数多くの選択肢があります。人気のモデルには、HuggingFace hubで利用できるモデルの[sentence-transformer](https://huggingface.co/models?library=sentence-transformers&sort=downloads)ファミリーや、[OpenAI embedding models](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings)などがあります:
# MAGIC
# MAGIC **注意** OpenAIEmbeddingsオブジェクトに必要なOpenAI API APIキーがnoteboo 00で設定され、[こちらの手順](https://python.langchain.com/en/latest/ecosystem/openai.html#installation-and-setup)に従って環境変数で利用できるようになっている必要があります。

# COMMAND ----------

# DBTITLE 1,ベクトルストアのロード
# エンべディングベクトルを生成するエンべディングモデルの指定
embeddings = OpenAIEmbeddings(model=config['openai_embedding_model'])

# ベクトルストアオブジェクトのインスタンスの作成
vector_store = FAISS.from_texts(
  embedding=embeddings, 
  texts=text_inputs, 
  metadatas=metadata_inputs
  )

# COMMAND ----------

# MAGIC %md 
# MAGIC 以降のノートブックでベクトルストアを利用できるようにストレージに永続化します：

# COMMAND ----------

# DBTITLE 1,ベクトルストアをストレージに永続化
vector_store.save_local(folder_path=config['vector_store_path'])

# COMMAND ----------

display(dbutils.fs.ls(config['vector_store_path'][5:]))

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | langchain | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/langchain/ |
# MAGIC | tiktoken | Fast BPE tokeniser for use with OpenAI's models | MIT  |   https://pypi.org/project/tiktoken/ |
# MAGIC | faiss-cpu | Library for efficient similarity search and clustering of dense vectors | MIT  |   https://pypi.org/project/faiss-cpu/ |
# MAGIC | openai | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/openai/ |
