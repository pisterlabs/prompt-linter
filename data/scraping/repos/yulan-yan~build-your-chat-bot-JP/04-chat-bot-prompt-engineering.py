# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Langchain and Dollyを活用したChat Bot 構築
# MAGIC
# MAGIC ## Chat Botのためのプロンプトエンジニアリング
# MAGIC
# MAGIC この例では、03の Q&A を改良してチャット ボットを作成します。
# MAGIC
# MAGIC 追加する主な点は、ボットが前のQ&A履歴をコンテキストとして持って回答できるように、前の質問と新しい質問の間に会話メモリを追加することです。
# MAGIC
# MAGIC
# MAGIC <img style="float:right" width="800px" src="https://github.com/yulan-yan/images-public/blob/6551b258815ed74ec5f54db34b6129e6325aa941/dolly_conversation.png?raw=true">
# MAGIC
# MAGIC ### 複数の質問の間で会話メモリを保持する
# MAGIC
# MAGIC チャット ボットの主な課題は、Q&A履歴全体をDollyに送信するコンテキストとして使用できないことです。
# MAGIC
# MAGIC なぜなら、まず第一に、これはコストがかかりますが、さらに重要なのは、モデル の最大ウィンドウ サイズよりも長いテキストになるため、Dollyでは長い議論がサポートされないことです。
# MAGIC
# MAGIC この課題を解決する秘訣は、自動要約モデルを使用し、Q＆A履歴の要約を取得してプロンプトに挿入する中間ステップを追加することです。
# MAGIC
# MAGIC これを行うには、「langchain」の「ConversationsummaryMemory」を使用して、中間要約タスクを追加します。
# MAGIC
# MAGIC
# MAGIC **注: これはより高度なコンテンツです。前のノートブック: 03-Q&A-prompt-engineering** から始めることをお勧めします。

# COMMAND ----------

# MAGIC %md
# MAGIC ### クラスターのセットアップ
# MAGIC
# MAGIC - Databricks Runtime 12.2 ML GPU を備えたクラスター上でこれを実行します。 13.0 ML GPU でも動作するはずです。
# MAGIC - このノートブックの例を実行するには、必要なのは GPU を備えた単一ノードの「クラスター」だけです
# MAGIC   - A100 インスタンスが最適に動作します。

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

# MAGIC %pip install -U transformers langchain chromadb accelerate bitsandbytes fugashi unidic-lite einops

# COMMAND ----------

# MAGIC %run ./_resources/00-init $catalog=hive_metastore $db=yyl

# COMMAND ----------

# DBTITLE 1,コンテキスト用のベクター データベースへ接続する
if len(get_available_gpus()) == 0:
  Exception("Running dolly without GPU will be slow. We recommend you switch to a Single Node cluster with at least 1 GPU to properly run this demo.")

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

dbqa_vector_db_path = "/dbfs"+demo_path+"/vector_db"
hf_embed = HuggingFaceEmbeddings(model_name="pkshatech/simcse-ja-bert-base-clcmlp")
chroma_db = Chroma(collection_name="dbqa_docs", embedding_function=hf_embed, persist_directory=dbqa_vector_db_path)

# COMMAND ----------

# DBTITLE 0,cvtbgeifduklhrn
# MAGIC %md 
# MAGIC ### 2/ 「langchain」とメモリによるプロンプト エンジニアリング
# MAGIC
# MAGIC これで、言語モデルとプロンプト戦略を組み合わせて、会話メモリを持って質問に答える「langchain」チェーンを作成できるようになりました。

# COMMAND ----------

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM, PreTrainedModel, PreTrainedTokenizer
from langchain import PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from langchain.memory import ConversationSummaryBufferMemory
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] ='0'


def build_qa_chain():
  torch.cuda.empty_cache()
  # Defining our prompt content.
  # langchain will load our similar documents as {context}
  template = """Below is an instruction with a question in it. Write a response that appropriately completes the request.

  ### Instruction:
  {context}

  {chat_history}

  ### Question:
  {human_input}

  ### Response:
  """
  prompt = PromptTemplate(input_variables=['context', 'human_input', 'chat_history'], template=template)

  # Increase max_new_tokens for a longer response
  # Other settings might give better results! Play around
  # local_output_dir = "/dbfs/dolly_training/dolly__2023-04-18T04:48:42"
  #model, tokenizer = load_model_tokenizer_for_generate(local_output_dir) 
  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  model = AutoModelForCausalLM.from_pretrained(
    'mosaicml/mpt-7b-instruct',
    trust_remote_code=True
  ).to(device)
  tokenizer = AutoTokenizer.from_pretrained('mosaicml/mpt-7b-instruct')
  
  # tokenizer = AutoTokenizer.from_pretrained(local_output_dir, padding_side="left")
  # model = AutoModelForCausalLM.from_pretrained(local_output_dir, device_map="auto", trust_remote_code=True)
  # inputs = tokenizer(prompt, return_tensors='pt').to(model.device) xx
  # input_length = inputs.input_ids.shape[1] xx
  end_key_token_id = tokenizer.encode("<|endoftext|>")[0]

  # instruct_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", pad_token_id=end_key_token_id, max_new_tokens=512, temperature=0.7, top_p=0.7, top_k=50, eos_token_id=end_key_token_id, return_dict_in_generate=True, device=device)
  instruct_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto", pad_token_id=end_key_token_id, max_new_tokens=512, top_p=0.7, top_k=50, eos_token_id=end_key_token_id, device=device)
  hf_pipe = HuggingFacePipeline(pipeline=instruct_pipeline)

  # Add a summarizer to our memory conversation
  # Let's make sure we don't summarize the discussion too much to avoid losing to much of the content

  # Models we'll use to summarize our chat history
  # We could use one of these models: https://huggingface.co/models?filter=summarization #facebook/bart-large-cnn
  summarize_model = AutoModelForSeq2SeqLM.from_pretrained("sonoisa/t5-base-japanese", device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)
  summarize_tokenizer = AutoTokenizer.from_pretrained("sonoisa/t5-base-japanese", padding_side="left")
  pipe_summary = pipeline("summarization", model=summarize_model, tokenizer=summarize_tokenizer) #, max_new_tokens=500, min_new_tokens=300
  # langchain pipeline doesn't support summarization yet, we added it as temp fix in the companion notebook _resources/00-init 
  hf_summary = HuggingFacePipeline_WithSummarization(pipeline=pipe_summary)
  #will keep 500 token and then ask for a summary. Removes prefix as our model isn't trained on specific chat prefix and can get confused.
  memory = ConversationSummaryBufferMemory(llm=hf_summary, memory_key="chat_history", input_key="human_input", max_token_limit=500, human_prefix = "", ai_prefix = "")

  # Set verbose=True to see the full prompt:
  print("loading chain, this can take some time...")
  return load_qa_chain(llm=hf_pipe, chain_type="stuff", prompt=prompt, verbose=True, memory=memory)

# COMMAND ----------

# MAGIC %md
# MAGIC ## チェーンを使用して簡単な質問に回答する
# MAGIC
# MAGIC それでおしまい！準備は完了です。質問に回答する関数を定義し、その回答を情報源とともにわかりやすく表示します。

# COMMAND ----------

class ChatBot():
  def __init__(self, db):
    self.reset_context()
    self.db = db

  def reset_context(self):
    self.sources = []
    self.discussion = []
    # Building the chain will load Dolly and can take some time depending on the model size and your GPU
    self.qa_chain = build_qa_chain()
    displayHTML("<h1>こんにちは！DatabricksのAIアシスタントです。何かご質問はありますか。</h1>")

  def get_similar_docs(self, question, similar_doc_count):
    return self.db.similarity_search(question, k=similar_doc_count)

  def chat(self, question):
    # Keep the last 3 discussion to search similar content
    self.discussion.append(question)
    similar_docs = self.get_similar_docs(" \n".join(self.discussion[-3:]), similar_doc_count=2)
    # Remove similar doc if they're already in the last questions (as it's already in the history)
    # similar_docs = [doc for doc in similar_docs if doc.metadata['source'] not in self.sources[-3:]]

    result = self.qa_chain({"input_documents": similar_docs, "human_input": question})
    # Cleanup the answer for better display:
    answer = result['output_text'].capitalize()
    result_html = f"<p><blockquote style=\"font-size:24\">{question}</blockquote></p>"
    result_html += f"<p><blockquote style=\"font-size:18px\">{answer}</blockquote></p>"
    result_html += "<p><hr/></p>"
    for d in result["input_documents"]:
      source_link = d.metadata["source"]
      self.sources.append(source_link)
      result_html += f"<p><blockquote>{d.page_content}<br/>(Source: <a href=\"{source_link}\">{source_link}</a>)</blockquote></p>"
    displayHTML(result_html)

chat_bot = ChatBot(chroma_db)

# COMMAND ----------

# MAGIC %md 
# MAGIC Databricksに関する質問をしてみます!

# COMMAND ----------

chat_bot.chat("AutoMLでサポートしている分類モデルのアルゴリズムは何ですか？")

# COMMAND ----------

chat_bot.chat("構築したモデルの予測に対して解釈できますか？")

# COMMAND ----------

chat_bot.chat("AutoMLで分類モデルを解釈できますか？")

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC ## 追加: MLFlow を使用して本番環境に langchain パイプラインをデプロイする (DBRML13以降が必要)
# MAGIC
# MAGIC ボットの準備ができたら、MLflow と langchain フレーバーを使用してパイプラインをパッケージ化できます。

# COMMAND ----------

# DBTITLE 1,Chat BotをMLflowにデプロイする
def publish_model_to_mlflow():
  # Build our langchain pipeline
  langchain_model = build_qa_chain()

  with mlflow.start_run() as run:
      # Save model to MLFlow
      # Note that this only saves the langchain pipeline (we could also add the ChatBot with a custom Model Wrapper class)
      # See https://mlflow.org/docs/latest/models.html#custom-python-models for an example
      # The vector database lives outside of your model
      mlflow.langchain.log_model(
          model=langchain_model,
          artifact_path="model"
      )
      model_registered = mlflow.register_model(f"runs:/{run.info.run_id}/model", "dbqa-bot-chain")

  # Move the model in production
  client = mlflow.tracking.MlflowClient()
  print("registering model version "+model_registered.version+" as production model")
  client.transition_model_version_stage("dbqa-bot", model_registered.version, stage = "Production", archive_existing_versions=True)

def load_model_and_answer(similar_docs, question): 
  # Load the langchain pipeline & run inferences
  chain = mlflow.pyfunc.load_model(model_uri)
  chain.predict({"input_documents": similar_docs, "human_input": question})

# COMMAND ----------

# publish_model_to_mlflow()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## チャットボットが完成しました！
# MAGIC
# MAGIC 以上で、チャットボットをデプロイする準備が整いました。
# MAGIC
# MAGIC ### まとめ
# MAGIC
# MAGIC このデモでは、会話履歴メモリを使用した基本的なプロンプト エンジニアリングをやってきました。これを元にしてより高度なソリューションを構築して、より適切なコンテキストを提供することが可能です。
# MAGIC
# MAGIC 高品質のトレーニング データセットを用意することが、モデルのパフォーマンスを向上させ、より適切なコンテキストを追加するための鍵となります。高品質のデータを収集して準備することが、ボットを成功させるために最も重要な部分と考えられます。
# MAGIC
# MAGIC データセットを改善する良い方法は、ユーザーの質問やチャットを収集し、Q&A データセットを段階的に改善することです。 <br/>
# MAGIC たとえば、「langchain」は、OpenAI と同様のデータセットでトレーニングされたチャット ボットとうまく動作するように開発されているので、Dolly のデータセットとは完全には一致しません。プロンプトがモデルのトレーニング データセットに一致するように設計されているほど、ボットの動作は向上します。
# MAGIC
# MAGIC *推論速度に関するメモ: 大きなモデルをロードすると、トランスフォーマー モデルのコンパイル時に推論時間を大幅に最適化できます。 onnx を使用した簡単な例を次に示します:*
# MAGIC
# MAGIC `%pip install -U transformers langchain chromadb accelerate bitsandbytes protobuf==3.19.0 optimum onnx onnxruntime-gpu`
# MAGIC
# MAGIC `%sh optimum-cli export onnx --model databricks/dolly-v2-7b  --device gpu --optimize O4 dolly_v2_7b_onnx`
# MAGIC
# MAGIC ```
# MAGIC from optimum.onnxruntime import ORTModelForCausalLM
# MAGIC
# MAGIC # Use Dolly as main model
# MAGIC model_name = "databricks/dolly-v2-3b" # can use dolly-v2-3b, dolly-v2-7b or dolly-v2-12b for smaller model and faster inferences.
# MAGIC tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
# MAGIC model = ORTModelForCausalLM.from_pretrained("databricks/dolly-v2-3b", export=True, provider="CUDAExecutionProvider")
# MAGIC ```
# MAGIC
# MAGIC *FasterTransformer を活用することもできます。詳細については、Databricks チームにお問い合わせください。*
