# Databricks notebook source
# MAGIC %md
# MAGIC # Dolly chat with langchain on Databricks
# MAGIC
# MAGIC Databricks' dolly-v2-7b, an instruction-following large language model trained on the Databricks machine learning platform that is licensed for commercial use. Based on pythia-6.9b, Dolly is trained on ~15k instruction/response fine tuning records databricks-dolly-15k generated by Databricks employees in capability domains from the InstructGPT paper, including brainstorming, classification, closed QA, generation, information extraction, open QA and summarization. dolly-v2-7b is not a state-of-the-art model, but does exhibit surprisingly high quality instruction following behavior not characteristic of the foundation model on which it is based.
# MAGIC
# MAGIC
# MAGIC Environment for this notebook:
# MAGIC - Runtime: 13.2 GPU ML Runtime
# MAGIC - Instance: g4dn.xlarge cluster (16gb, 4 cores) on AWS
# MAGIC
# MAGIC ## What is Langchain?
# MAGIC
# MAGIC LangChain is an intuitive open-source Python framework build automation around LLMs), and allows you to build dynamic, data-responsive applications that harness the most recent breakthroughs in natural language processing.

# COMMAND ----------

# Huggingface login not needed since open model
# from huggingface_hub import notebook_login

# # Login to Huggingface to get access to the model
# notebook_login()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Inference
# MAGIC The example in the model card should also work on Databricks with the same environment.
# MAGIC
# MAGIC Takes about 8m on g4dn.xlarge cluster (16gb, 4 cores).

# COMMAND ----------

import torch
from transformers import pipeline

generate_text = pipeline(model="databricks/dolly-v2-7b", torch_dtype=torch.bfloat16, 
                         revision='d632f0c8b75b1ae5b26b250d25bfba4e99cb7c6f',
                         trust_remote_code=True, device_map="auto", return_full_text=True)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC You can create a prompt that either has only an instruction or has an instruction with context:

# COMMAND ----------

from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline

# template for an instrution with no input
prompt = PromptTemplate(
    input_variables=["instruction"],
    template="{instruction}")

# template for an instruction with input
prompt_with_context = PromptTemplate(
    input_variables=["instruction", "context"],
    template="{instruction}\n\nInput:\n{context}")

hf_pipeline = HuggingFacePipeline(pipeline=generate_text)

llm_chain = LLMChain(llm=hf_pipeline, prompt=prompt)
llm_context_chain = LLMChain(llm=hf_pipeline, prompt=prompt_with_context)

# COMMAND ----------

# MAGIC %md
# MAGIC Example predicting using a simple instruction:

# COMMAND ----------

print(llm_chain.predict(instruction="Explain to me the difference between nuclear fission and fusion.").lstrip())

# COMMAND ----------

context = """George Washington (February 22, 1732[b] - December 14, 1799) was an American military officer, statesman,
and Founding Father who served as the first president of the United States from 1789 to 1797."""

print(llm_context_chain.predict(instruction="When was George Washington president?", context=context).lstrip())

# COMMAND ----------

print(llm_chain.predict(instruction="When was George Washington president?").lstrip())

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Non-context Example

# COMMAND ----------

print(llm_chain.predict(instruction="What determines how fast you can reply to the requests?").lstrip())

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Context Example

# COMMAND ----------

context = """Haakon IV Haakonsson, byname Haakon The Old, Norwegian Håkon Håkonsson, or Håkon Den Gamle, (born 1204, Norway—died December 1263, Orkney Islands), king of Norway (1217–63) who consolidated the power of the monarchy, patronized the arts, and established Norwegian sovereignty over Greenland and Iceland. His reign is considered the beginning of the “golden age” (1217–1319) in medieval Norwegian history."""

print(llm_context_chain.predict(instruction="What characterized Haakon IV Haakonson?", context=context).lstrip())

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Task:
# MAGIC
# MAGIC Play around with the context and the instructions.

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Task: Get a good answer about Knowit

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Get an even better answer by providing context

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Task: Get a good description of Oslo

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ## Task: Try to get a description of Oslo from a French perspective (French people, not French language)

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Task: Explore advanced usage of LangChain
# MAGIC
# MAGIC https://github.com/gkamradt/langchain-tutorials/blob/main/LangChain%20Cookbook%20Part%202%20-%20Use%20Cases.ipynb

# COMMAND ----------


