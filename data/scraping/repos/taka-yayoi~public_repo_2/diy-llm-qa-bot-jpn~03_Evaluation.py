# Databricks notebook source
# MAGIC %md このノートブックの目的は、QA Botアクセラレータを構成するノートブックを制御するさまざまな設定値を設定することです。このノートブックは https://github.com/databricks-industry-solutions/diy-llm-qa-bot から利用できます。

# COMMAND ----------

# MAGIC %md ## QAモデルのパフォーマンスの評価
# MAGIC
# MAGIC このノートブックでは、langchainの`QAEvalChain`と正しいリファレンスとレスポンスを含む評価セットを用いてどのようにQAのパフォーマンスを評価するのかを説明します。モデルのレスポンスと正しいレスポンスを比較するために、試験官としてLLMを活用します。

# COMMAND ----------

# MAGIC %run "./util/notebook-config"

# COMMAND ----------

# MAGIC %md 
# MAGIC ## 評価セットのスコアリング
# MAGIC
# MAGIC モデルのスコアリングするために、記録されたモデルからPythonの依存関係を取得してインストールします。
# MAGIC
# MAGIC MLflowはDBFSの`requirements_path`にモデルの依存関係を含むファイルを書き込みます。そして、ファイルの依存関係をインストールするために %pip を使います。

# COMMAND ----------

import mlflow

requirements_path = mlflow.pyfunc.get_model_dependencies(config['model_uri'])
%pip install -r $requirements_path
dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md 
# MAGIC 上のブロックはカーネルを再起動するので、再度設定を行い依存関係をインポートします:

# COMMAND ----------

# MAGIC %run "./util/notebook-config"

# COMMAND ----------

import pandas as pd
import numpy as np
import openai
import json
from langchain.llms import OpenAI
import os
from langchain.evaluation.qa import QAEvalChain
import mlflow

# COMMAND ----------

# MAGIC %md 
# MAGIC 質問と、正しいリファレンスとサンプルの回答の両方を含む正しい回答の評価セットを準備しました。サンプルを見てみましょう：

# COMMAND ----------

eval_dataset = pd.read_csv("/Workspace/Users/takaaki.yayoi@databricks.com/20230525_diy-llm-qa-bot/data/eval_file.tsv", sep='\t').to_dict('records')
#eval_dataset = pd.read_csv(config['eval_dataset_path'], sep='\t').to_dict('records')
eval_dataset[0] 

# COMMAND ----------

# DBTITLE 1,記録されたモデルによる評価データセットのスコアリング
queries = pd.DataFrame({'question': [r['question'] for r in eval_dataset]})
model = mlflow.pyfunc.load_model(config['model_uri'])
predictions = model.predict(queries)
predictions[0]

# COMMAND ----------

# MAGIC %md 
# MAGIC langchainの`QAEvalChain`は試験官として動作します： それぞれの質問に対して、スコアリングされた回答が正解データと十分類似しているかどうかを比較し、CORRECTかINCORRECTを返却します。

# COMMAND ----------

llm = OpenAI(temperature=0)
eval_chain = QAEvalChain.from_llm(llm)
graded_outputs = eval_chain.evaluate(eval_dataset, predictions, question_key="question", prediction_key="answer")
graded_outputs[:5]

# COMMAND ----------

# MAGIC %md 
# MAGIC 評価された結果をコンパイルし、評価された回答と評価データセットを一つのデータフレームにすることができます。botは時には、別の参照ソースをベースとして正しい回答を生成することがあることに注意してください。

# COMMAND ----------

results = pd.DataFrame(
  [{
    "question": eval_data["question"], 
    "prediction": predict["answer"], 
    "source": predict["source"], 
    "correct_source": eval_data["correct_source"], 
    "answer": eval_data["answer"], 
    "find_correct_source": predict["source"] == eval_data["correct_source"], 
    "same_as_answer": True if graded_output['text'].strip() == 'CORRECT' else False
    } 
    for (predict, eval_data, graded_output) in zip(predictions, eval_dataset, graded_outputs)])
display(spark.createDataFrame(results))

# COMMAND ----------

# DBTITLE 1,モデルは評価データセットに対してどのくらい正しい回答をするのでしょうか？
results['same_as_answer'].sum() / len(results)

# COMMAND ----------

# MAGIC %md 
# MAGIC LLMグレーダーによると、我々のQA botは多くの場合で合理的なレスポンスをしているようです。しかし、開発者はレスポンスを読むことで、パフォーマンスを定期的に評価することが依然として重要です。特に長くて複雑な質問の場合には、LLMが微妙なコンセプトの違いを見逃して、偽陰性の評価を行うことははよくあることです。そして、ユーザーが送信する質問のタイプが反映されるように、定期的に評価質問セットをレビューするようにしてください。

# COMMAND ----------

# MAGIC %md © 2023 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | langchain | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/langchain/ |
# MAGIC | tiktoken | Fast BPE tokeniser for use with OpenAI's models | MIT  |   https://pypi.org/project/tiktoken/ |
# MAGIC | faiss-cpu | Library for efficient similarity search and clustering of dense vectors | MIT  |   https://pypi.org/project/faiss-cpu/ |
# MAGIC | openai | Building applications with LLMs through composability | MIT  |   https://pypi.org/project/openai/ |

# COMMAND ----------


