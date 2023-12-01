# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # LangchainとDollyを活用したカスタム データセットでの質問回答
# MAGIC
# MAGIC ## プロンプトエンジニアリング
# MAGIC
# MAGIC プロンプト エンジニアリングは、特定のユーザーの質問をより多くの情報で包み込み、より適切な回答を得るためにモデルをガイドする手法です。<br/>
# MAGIC プロンプト エンジニアリングには通常、次の内容が含まれます。
# MAGIC - 用途を応じて回答方法のガイダンス (*例: あなたは庭師です。植物を生かしておくために、次の質問にできる限り最善を尽くして答えてください*)
# MAGIC - 高品質な回答を得るためのコンテキストの追加。たとえば、ユーザーの質問に近いテキスト (*例: [社内 Q&A の内容] を承知の上で、回答してください...*)
# MAGIC - 回答に対して具体的な指示 (*例: イタリア語での回答*)
# MAGIC - チャット ボットを構築している場合に以前のやり取りをコンテキストとして保持する (埋め込みとして圧縮)
# MAGIC - ...
# MAGIC
# MAGIC <img style="float:right" width="700px" src="https://github.com/yulan-yan/images-public/blob/6551b258815ed74ec5f54db34b6129e6325aa941/dolly_QAinference.png?raw=true">
# MAGIC
# MAGIC この例では、より良いプロンプトを作成するために「langchain」を使用します。
# MAGIC
# MAGIC ## AIアシスタントのプロンプトエンジニアリング
# MAGIC
# MAGIC この例では、「langchain」、Hugging Face「transformers」、さらには Apache Spark を適用して、特定のテキスト コーパスに関する質問に答える方法を示します。
# MAGIC
# MAGIC mosaicml/mpt-7b-instruct LLM を使用していますが、この例では任意のテキスト生成 LLM や、若干の変更を加えれば OpenAI も使用できます。
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
# MAGIC 以下のようなフローを実装していきます。 <br><br>
# MAGIC
# MAGIC <div style="margin-left: 20px">
# MAGIC   <div class="badge_b"><div class="badge">1</div> ユーザーからの質問をインプットし、同じセンテンステキストをベクトル化するモデルを使用して埋め込みとして変換します。</div>
# MAGIC   <div class="badge_b"><div class="badge">2</div> Chroma内で類似検索を実行して、関連する質問と回答を見つけます。</div>
# MAGIC   <div class="badge_b"><div class="badge">3</div> 質問と関連するQ&Aをコンテキストとして含んたプロンプトを加工します。</div>
# MAGIC   <div class="badge_b"><div class="badge">4</div> プロンプトをDollyに送信します。</div>
# MAGIC   <div class="badge_b"><div class="badge">5</div> ユーザーは回答を受けます。</div>
# MAGIC </div>
# MAGIC <br/>

# COMMAND ----------

!wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusparse-dev-11-3_11.5.0.58-1_amd64.deb -O /tmp/libcusparse-dev-11-3_11.5.0.58-1_amd64.deb && \
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcublas-dev-11-3_11.5.1.109-1_amd64.deb -O /tmp/libcublas-dev-11-3_11.5.1.109-1_amd64.deb && \
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcusolver-dev-11-3_11.1.2.109-1_amd64.deb -O /tmp/libcusolver-dev-11-3_11.1.2.109-1_amd64.deb && \
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/libcurand-dev-11-3_10.2.4.109-1_amd64.deb -O /tmp/libcurand-dev-11-3_10.2.4.109-1_amd64.deb && \
  dpkg -i /tmp/libcusparse-dev-11-3_11.5.0.58-1_amd64.deb && \
  dpkg -i /tmp/libcublas-dev-11-3_11.5.1.109-1_amd64.deb && \
  dpkg -i /tmp/libcusolver-dev-11-3_11.1.2.109-1_amd64.deb && \
  dpkg -i /tmp/libcurand-dev-11-3_10.2.4.109-1_amd64.deb

# COMMAND ----------

# MAGIC %pip install -U transformers langchain chromadb accelerate bitsandbytes fugashi unidic-lite einops SentencePiece

# COMMAND ----------

# %pip install -U accelerate==0.16.0 click==8.0.4 datasets==2.10.0 deepspeed==0.8.3 transformers[torch]==4.28.1 langchain==0.0.139 torch==1.13.1

# COMMAND ----------

# MAGIC %run ./_resources/00-init $catalog=hive_metastore $db=yyl

# COMMAND ----------

# MAGIC %md
# MAGIC ### クラスターのセットアップ
# MAGIC
# MAGIC - Databricks Runtime 12.2 ML GPU を備えたクラスター上でこれを実行します。 13.0 ML GPU でも動作するはずです。
# MAGIC - このノートブックの例を実行するに必要なのは GPU を備えた単一ノードの「クラスター」だけです
# MAGIC   - 小さなモデルを使う場合はA10 および V100 インスタンスは動作するものがあります。
# MAGIC   - A100 インスタンスが最適に動作します。

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 1/ Hugging Faceから埋め込みモデルをダウンロードします (データ準備と同じ)。　

# COMMAND ----------

# Start here to load a previously-saved DB
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

if len(get_available_gpus()) == 0:
  Exception("Running dolly without GPU will be slow. We recommend you switch to a Single Node cluster with at least 1 GPU to properly run this demo.")

dbqa_vector_db_path = "/dbfs"+demo_path+"/vector_db"

hf_embed = HuggingFaceEmbeddings(model_name="pkshatech/simcse-ja-bert-base-clcmlp")
db = Chroma(collection_name="dbqa_docs", embedding_function=hf_embed, persist_directory=dbqa_vector_db_path)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 2/ Chromaを利用した類似検索
# MAGIC
# MAGIC 簡単な質問で類似検索をテストしてみましょう。
# MAGIC
# MAGIC `k` (`similar_doc_count`)は、プロンプトに送信するために取得されたテキストのチャンクの数であることに注意してください。プロンプトが長いと、より多くのコンテキストが追加されますが、処理に時間がかかります。

# COMMAND ----------

def get_similar_docs(question, similar_doc_count):
  return db.similarity_search(question, k=similar_doc_count)

# Let's test it with blackberries:
for doc in get_similar_docs("AutoMLでモデルを解釈できますか？", 1):
  print(doc.page_content)

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 3/ 「langchain」によるプロンプト エンジニアリング
# MAGIC
# MAGIC これで、言語モデルとプロンプト戦略を組み合わせて、質問に答える「langchain」チェーンを作成できるようになりました。

# COMMAND ----------

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, PreTrainedModel, PreTrainedTokenizer
import torch
from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain

import numpy as np

def build_qa_chain():
  torch.cuda.empty_cache()

  # Defining our prompt content.
  # langchain will load our similar documents as {context}
  template = """Below is an instruction with a question in it. Write a response that appropriately completes the request.

  ### Instruction:
  {context}

  ### Question:
  {question}

  ### Response:
  """
  prompt = PromptTemplate(input_variables=['context', 'question'], template=template)

  # Load model and build pipeline
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = AutoModelForCausalLM.from_pretrained(
    'mosaicml/mpt-7b-instruct',
    trust_remote_code=True
  ).to(device)
  tokenizer = AutoTokenizer.from_pretrained('mosaicml/mpt-7b-instruct')

  end_key_token_id = tokenizer.encode("<|endoftext|>")[0]

  # Increase max_new_tokens for a longer response
  # Other settings might give better results! Play around
  instruct_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", pad_token_id=end_key_token_id, max_new_tokens=512, top_p=0.7, top_k=50, eos_token_id=end_key_token_id, device=device)
  # Note: if you use dolly 12B or smaller model but a GPU with less than 24GB RAM, use 8bit. This requires %pip install bitsandbytes
  # instruct_pipeline = pipeline(model=model_name, load_in_8bit=True, trust_remote_code=True, device_map="auto")
  # For GPUs without bfloat16 support, like the T4 or V100, use torch_dtype=torch.float16 below
  
  hf_pipe = HuggingFacePipeline(pipeline=instruct_pipeline)

  # Set verbose=True to see the full prompt:
  return load_qa_chain(llm=hf_pipe, chain_type="stuff", prompt=prompt, verbose=True)

# Building the chain will load Dolly and can take several minutes depending on the model size
qa_chain = build_qa_chain()

# COMMAND ----------

# MAGIC %md
# MAGIC 言語モデルが質問にどのように答えるかに影響を与える要因は多数あることに注意してください。最も注目されるのは、プロンプト テンプレート自体です。これは変更することができ、特定のモデルでは異なるプロンプトがより良く機能する場合もあれば、より悪く機能する場合もあります。
# MAGIC
# MAGIC 回答生成プロセス自体にもチューニングすべき多くのノブがあり、多くの場合、特定のモデルや特定のデータセットに最適な設定を見つけるために単に試行錯誤が必要になります。こちら [Hugging Face の優れたガイド](https://huggingface.co/blog/how-to-generate) を参照してください。
# MAGIC
# MAGIC パフォーマンスに最も影響を与える設定は次のとおりです。
# MAGIC - `max_new_tokens`: 応答が長いほど、生成に時間がかかります。短くすると応答が短くなり、応答が速くなります。
# MAGIC - `num_beams`: beam検索を使用する場合、beamの数が増えると実行時間は多かれ少なかれ直線的に増加します。

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4/ チェーンを使用して簡単な質問に回答する
# MAGIC
# MAGIC それでおしまい！準備は完了です。質問に回答する関数を定義し、その回答を情報源とともにわかりやすく表示します。

# COMMAND ----------

def answer_question(question):
  similar_docs = get_similar_docs(question, similar_doc_count=2)
  result = qa_chain({"input_documents": similar_docs, "question": question})
  result_html = f"<p><blockquote style=\"font-size:24\">{question}</blockquote></p>"
  result_html += f"<p><blockquote style=\"font-size:18px\">{result['output_text']}</blockquote></p>"
  result_html += "<p><hr/></p>"
  for d in result["input_documents"]:
    source_id = d.metadata["source"]
    result_html += f"<p><blockquote>{d.page_content}<br/>(Source: <a href=\"https://gardening.stackexchange.com/a/{source_id}\">{source_id}</a>)</blockquote></p>"
  displayHTML(result_html)

# COMMAND ----------

# MAGIC %md 
# MAGIC Databricksに関する質問をしてみます!

# COMMAND ----------

answer_question("AutoMLで分類モデルを説明できますか？")

# COMMAND ----------

answer_question("AutoMLでサポートしている分類モデルのアルゴリズムは何ですか？")

# COMMAND ----------

answer_question("内部結合 (INNER JOIN)の際に使うメソッドは何ですか?")

# COMMAND ----------

answer_question("AutoMLではFeature Storeのテーブルをサポートしていますか？")

# COMMAND ----------

# MAGIC %md ## その他

# COMMAND ----------

questions = [
    "AutoMLでモデルを解釈できますか？",
    "データレイクハウスとは何ですか?",
    "AutoMLで回帰モデルを作れますか？",
    "AutoMLではFeature Storeのテーブルをサポートしていますか？",
    "分散機械学習をどのように行うか?",
    "データレイクハウスとデータウェアハウスの違いは?",
    "MLOpsのベストプラクティスについて教えてください。",
    "内部結合 (INNER JOIN)の際に使うメソッドは何ですか？",
    "2つのデータフレームをどのように縦方向に結合するか？",
    "データプレーンとコントロールプレーンの違いは何?",
    "Databricksでデータウェアハウスをどのように構築するか？",
    "特徴量の再利用をどのように実現するか？",
    "LangChainで作ったモデルをMLflowで管理できるか？",
    "Snowflakeと比較してDatabricksのメリットは何ですか?",
    "Pandasデータフレームとsparkデータフレームの違いは何？",
    "データプレーンはどこに配置されるか?",
]

# 各質問に対して応答を生成して表示
for question in questions:
  answer_question(question)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC # 次へ： チャットボットとして質問を連鎖させるための Q&A プロンプトを改善する
# MAGIC
# MAGIC 次のノートブック [04-chat-bot-prompt-engineering]($./04-chat-bot-prompt-engineering) を開いて、チェーンを改善し、やり取りをプロンプトに追加します。

# COMMAND ----------


