# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Dolly: Databricks Lakehouseによるデータ準備 & ベクトルデータベース作成
# MAGIC
# MAGIC <img style="float: right" width="600px" src="https://github.com/yulan-yan/images-public/blob/6551b258815ed74ec5f54db34b6129e6325aa941/dolly_dataPrep.png?raw=true">
# MAGIC
# MAGIC モードを特化できるようにするには、トレーニング データセットとして使用する Q&A のリストが必要です。
# MAGIC
# MAGIC このデモでは、手動で収集したDatabricksのQ＆A データセットを使用してモデルを特化します。
# MAGIC
# MAGIC DatabricksのQ＆A データセットを取り込み、トレーニングのために保存する単純なデータ パイプラインから始めましょう。
# MAGIC
# MAGIC 次の手順で実装します。 <br><br>
# MAGIC
# MAGIC <style>
# MAGIC .right_box{
# MAGIC   margin: 30px; box-shadow: 10px -10px #CCC; width:650px; height:300px; background-color: #1b3139ff; box-shadow:  0 0 10px  rgba(0,0,0,0.6);
# MAGIC   border-radius:25px;font-size: 35px; float: left; padding: 20px; color: #f9f7f4; }
# MAGIC .badge {
# MAGIC   clear: left; float: left; height: 30px; width: 30px;  display: table-cell; vertical-align: middle; border-radius: 50%; background: #fcba33ff; text-align: center; color: white; margin-right: 10px; margin-left: -35px;}
# MAGIC .badge_b { 
# MAGIC   margin-left: 25px; min-height: 32px;}
# MAGIC </style>
# MAGIC
# MAGIC <div style="margin-left: 20px">
# MAGIC   <div class="badge_b"><div class="badge">1</div> Q&A データセット(jsonファイル)をアップロードする</div>
# MAGIC   <div class="badge_b"><div class="badge">2</div> 質問と回答のペアの文書を準備する</div>
# MAGIC   <div class="badge_b"><div class="badge">3</div> センテンステキストをベクトル化するモデルを活用して文書をベクトル化する</div>
# MAGIC   <div class="badge_b"><div class="badge">4</div> ベクトルデータベース（Chroma）にベクトルをインデックス付ける</div>
# MAGIC </div>
# MAGIC <br/>

# COMMAND ----------

# DBTITLE 1,ベクトルデータベースをインストールする
# MAGIC %pip install -U chromadb langchain transformers fugashi unidic-lite

# COMMAND ----------

# MAGIC %run ./_resources/00-init $catalog=hive_metastore $db=yyl

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1/ 生データのjsonファイルをアップロードする
# MAGIC
# MAGIC 手動作成した質問回答のペアのデータをDatabricksのDBFSにアップロードします。

# COMMAND ----------

dbqa_df = spark.read.option("multiline","true").json("/FileStore/Users/yulan.yan/qa_dataset/dbqa_public.json")

display(dbqa_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 2/ 質問と回答のペアを結合して一つの列に入れる
# MAGIC
# MAGIC 結果をデルタテーブルに保存します。

# COMMAND ----------

all_texts = dbqa_df.select(col("source"), concat(col("instruction"), lit("\n\n"), col("response"))).toDF("source", "text")
spark.sql('drop table if exists yyl.dolly_dbqa_source_texts')
all_texts.write.format("delta").mode("overwrite").saveAsTable("yyl.dolly_dbqa_source_texts")

# COMMAND ----------

# DBTITLE 1,質問と回答のペアを一つの列に入れたデータを読み込む
        
# Prepare the training dataset: question following with the best answers.
docs_df = spark.table("yyl.dolly_dbqa_source_texts").filter(col("text").isNotNull()) 
display(docs_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### 推論を高速化するために短いバージョンを追加する
# MAGIC
# MAGIC Q＆Aデータセットは、質問とそれに続く回答で構成されています。
# MAGIC
# MAGIC 潜在的な問題は、これがかなり長いテキストになる可能性があることです。長いテキストをコンテキストとして使用すると、LLM推論が遅くなる可能性があります。 1つのオプションは、自動要約LLMを使用してこれらの Q&A を要約し、結果を新しいフィールドとして保存することです。
# MAGIC
# MAGIC この操作には時間がかかる場合があります。そのため、推論中にQ&Aを要約する必要がないように、データ準備パイプラインでこの操作を1回のみ実行します。

# COMMAND ----------

# DBTITLE 1,テキストの自動予約を追加する
from typing import Iterator
import pandas as pd 
from transformers import pipeline

@pandas_udf("string")
def summarize(iterator: Iterator[pd.Series]) -> Iterator[pd.Series]:
    # Load the model for summarization
    summarizer = pipeline("summarization", model="sonoisa/t5-base-japanese")
    def summarize_txt(text):
      if len(text) > 400:
        return summarizer(text)
      return text

    for serie in iterator:
        # get a summary for each row
        yield serie.apply(summarize_txt)

# We won't run it as this can take some time in the entire dataset
# docs_df = docs_df.withColumn("text_short", summarize("text"))
docs_df.write.mode("overwrite").saveAsTable(f"dbqa_dataset")
display(docs_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 3/ モデルをロードしてドキュメントを埋め込みベクトルに変換します
# MAGIC
# MAGIC 簡単にHuggingfaceからセンテンステキストを埋め込みベクトルに変換するモデルをロードし、後で chromadb クライアントで使用します。

# COMMAND ----------

from langchain.embeddings import HuggingFaceEmbeddings

# Download model from Hugging face
hf_embed = HuggingFaceEmbeddings(model_name="pkshatech/simcse-ja-bert-base-clcmlp")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 4/ ベクトルデータベース内のドキュメント (行) にインデックスを付ける
# MAGIC
# MAGIC 次に、準備したテキストをロードし、「langchain」パイプラインで使用するための検索可能なテキストのデータベースを作成します。 
# MAGIC <br>
# MAGIC これらのドキュメントは埋め込みベクトルに変更されているため、後のクエリもベクトルに変更し、ベクトルによって関連するテキスト チャンクとマッチングすることができます。
# MAGIC
# MAGIC - Spark を使用してテキスト チャンクを収集します。 「langchain」は、Word ドキュメント、GDrive、PDF などから直接チャンクを読み取ることもサポートしています。
# MAGIC - ストレージに保存できる簡単なインメモリ Chroma ベクトル データベース を作成します。
# MAGIC - `sentence-transformers` から埋め込み関数をインスタンス化します。
# MAGIC - データベースにデータを入力して保存します。

# COMMAND ----------

# MAGIC %sh 
# MAGIC rm -r /dbfs/Users/yulan.yan@databricks.com/llm/dbqa/vector_db
# MAGIC mkdir -p /dbfs/Users/yulan.yan@databricks.com/llm/dbqa/vector_db

# COMMAND ----------

# DBTITLE 1,ベクトルデータベースの保存場所を用意する (dbfsに)
# Prepare a directory to store the document database. Any path on `/dbfs` will do.
dbutils.widgets.dropdown("reset_vector_database", "false", ["false", "true"], "Recompute embeddings for chromadb")
dbqa_vector_db_path = "/dbfs"+demo_path+"/vector_db"

# Don't recompute the embeddings if the're already available
compute_embeddings = dbutils.widgets.get("reset_vector_database") == "true" or is_folder_empty(dbqa_vector_db_path)

if compute_embeddings:
  print(f"creating folder {dbqa_vector_db_path} under our blob storage")
  dbutils.fs.rm(dbqa_vector_db_path, True)
  dbutils.fs.mkdirs(dbqa_vector_db_path)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ドキュメント データベースを作成します。
# MAGIC - 比較的小さなテキストデータセットを収集し、`Document`を作成するだけです。 `langchain` は、PDF、GDrive ファイルなどから直接ドキュメント コレクションを形成することもできます。
# MAGIC - 長いテキストを管理しやすいチャンクに分割する (オプション)

# COMMAND ----------

from langchain.docstore.document import Document
from langchain.vectorstores import Chroma

all_texts = spark.table("dbqa_dataset")

print(f"Saving document embeddings under {dbqa_vector_db_path}")

if compute_embeddings: 
  # Transform our rows as langchain Documents
  # If you want to index shorter term, use the text_short field instead
  documents = [Document(page_content=r["text"], metadata={"source": r["source"]}) for r in all_texts.collect()]

  # If your texts are long, you may need to split them:
  # text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100)
  # documents = text_splitter.split_documents(documents)

  # Init the chroma db with the pkshatech/simcse-ja-bert-base-clcmlp model loaded from hugging face  (hf_embed)
  db = Chroma.from_documents(collection_name="dbqa_docs", documents=documents, embedding=hf_embed, persist_directory=dbqa_vector_db_path)
  db.similarity_search("dummy") # tickle it to persist metadata (?)
  db.persist()

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## 以上で、Q&A データセットの準備が整いました。
# MAGIC
# MAGIC このノートブックでは、Databricks を利用して Q&A データセットを準備します。
# MAGIC
# MAGIC * データセットの取り込み
# MAGIC * 埋め込みを準備してchromaに保存する
# MAGIC
# MAGIC このデータセットを使用してプロンプト コンテキストを改善し、Databricks AI アシスタントを構築する準備が整いました。
# MAGIC 次のノートブックを開きます [03-Q&A-prompt-engineering]($./03-Q&A-prompt-engineering)
