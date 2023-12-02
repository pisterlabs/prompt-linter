# Databricks notebook source
# MAGIC %md The purpose of this notebook is to define and persist the model to be used by the QA Bot accelerator.  This notebook is inspired from https://github.com/databricks-industry-solutions/diy-llm-qa-bot.

# COMMAND ----------

# MAGIC %md ##Introduction
# MAGIC
# MAGIC In this notebook, we'll first walk through these steps one at a time so that we can wrap our head around what all is taking place.  We will then repackage the logic as a class object which will allow us to more easily encapsulate our work.  We will persist that object as a model within MLflow which will assist us in deploying the model in the last notebook associated with this accelerator.

# COMMAND ----------

# DBTITLE 1,Install Required Libraries
# MAGIC %run "../../databricks_llm/prompt_utils/install-prep-libraries"

# COMMAND ----------

# DBTITLE 1,Optional : Load Ray Dashboard to show cluster Utilisation
# MAGIC %run "../../databricks_llm/prompt_utils/install_ray"

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
import re
import time
import pandas as pd
import mlflow
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.schema import BaseRetriever
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts import PromptTemplate
from langchain.base_language import BaseLanguageModel

from langchain import LLMChain
from datasets import Dataset, load_dataset
from databricks_llm.prompt_utils.mptbot import (
    HuggingFacePipelineLocal,
    TGILocalPipeline,
)

# COMMAND ----------

# DBTITLE 1,Get Config Settings
# MAGIC %run "../../databricks_llm/prompt_utils/notebook-config"

# COMMAND ----------

# MAGIC %md ##Step 1: Explore entity creation
# MAGIC
# MAGIC To get started, let's explore how we will derive an answer in response to a user provide question.  We'll start by defining that question here:

# COMMAND ----------

# DBTITLE 1,Specify Question
dataset = load_dataset(
    config["hf_dataset"],
    split="train",
    cache_dir="/tmp/hf/cache",
    data_dir="/tmp/hf/data",
)

dataset = dataset.select(range(min(len(dataset), 50)))

# COMMAND ----------

dataset = dataset.to_pandas()

# COMMAND ----------

print("meaning_representation:", dataset.iloc[-3]["meaning_representation"])
print("human_reference:", dataset.iloc[-3]["human_reference"])

# COMMAND ----------

# MAGIC %md Using our vector store, assembled in the prior notebook, we will retrieve document chunks relevant to the question:
# MAGIC
# MAGIC **NOTE** The OpenAI API key used by the OpenAIEmbeddings object is specified in an environment variable set during the earlier `%run` call to get configuration variables.

# COMMAND ----------

# DBTITLE 1,Create Correct Prompt Structure
config[
    "template"
] = """<|im_start|>system\n- You are an assistant which helps extracts important entities.If the input is incorrect just say I cannot answer the questions.Do not add entities which are not explicitly mentioned \n<|im_end|>\n<|im_start|>user\n text: {context} \n <|im_end|><|im_start|>\n assistant""".strip().strip()

# COMMAND ----------

# MAGIC %md We can now turn our attention to the prompt that we will send to the model.  This prompt needs to include placeholders for the *question* the user will submit and the document that we believe will provide the *context* for answering it.
# MAGIC
# MAGIC Please note that the prompt consists multiple prompt elements, defined using [prompt templates](https://python.langchain.com/en/latest/modules/prompts/chat_prompt_template.html).  In a nutshell, prompt templates allow us to define the basic structure of a prompt and more easily substitute variable data into them to trigger a response.  The system message prompt shown here provides instruction to the model about how we want it to respond.  The human message template provides the details about the user-initiated request.
# MAGIC
# MAGIC The prompts along with the details about the model that will respond to the prompt are encapsulated within an [LLMChain object](https://python.langchain.com/en/latest/modules/chains/generic/llm_chain.html).  This object simply defines the basic structure for resolving a query and returning a reponse:

# COMMAND ----------

dataset.iloc[0]


# COMMAND ----------

# DBTITLE 1,Define Chain to Generate Responses
# define system-level instructions
system_message_prompt = SystemMessagePromptTemplate.from_template(config["template"])
chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])


if config["model_id"] == "openai":
    # define model to respond to prompt
    llm = ChatOpenAI(
        model_name=config["openai_chat_model"], temperature=config["temperature"]
    )

else:
    # define model to respond to prompt
    llm = TGILocalPipeline.from_model_id(
        model_id=config["model_id"],
        model_kwargs=config["model_kwargs"],
        pipeline_kwargs=config["pipeline_kwargs"],
    )

# combine prompt and model into a unit of work (chain)
qa_chain = LLMChain(llm=llm, prompt=chat_prompt)

# COMMAND ----------

# MAGIC %md To actually trigger a response, we will loop through each of our docs from highest to lowest relevance and attempt to elicit a response.  Once we get a valid response, we'll stop.
# MAGIC
# MAGIC Please note, we aren't providing time-out handling or thoroughly validating the response from the model in this next cell.  We will want to make this logic more robust as we assemble our application class but for now we'll keep it simple to ensure the code is easy to read:

# COMMAND ----------

# DBTITLE 1,Generate a Response
# for each provided document
text = dataset.iloc[10]["human_reference"]

# generate a response
output = qa_chain.generate([{"context": text}])

# get answer from results
generation = output.generations[0][0]
answer = generation.text

# display answer
if answer is not None:
    print(f"Context: {text}", "\n", f"Answer: {answer}")

# COMMAND ----------

# MAGIC %md ##Step 3: Persist Model to MLflow
# MAGIC
# MAGIC With our bot class defined and validated, we can now persist it to MLflow.  MLflow is an open source repository for model tracking and logging.  It's deployed by default with the Databricks platform, making it easy for us to record models with it.
# MAGIC
# MAGIC While MLflow now [supports](https://www.databricks.com/blog/2023/04/18/introducing-mlflow-23-enhanced-native-llm-support-and-new-features.html) both OpenAI and LangChain model flavors, the fact that we've written custom logic for our bot application means that we'll need to make use of the more generic [pyfunc](https://mlflow.org/docs/latest/python_api/mlflow.pyfunc.html#creating-custom-pyfunc-models) model flavor.  This model flavor allows us to write a custom wrapper for our model that gives us considerable control over how our model responds when deployed through standard, MLflow-provided deployment mechanisms.
# MAGIC
# MAGIC To create a custom MLflow model, all we need to do is define a class wrapper of type *mlflow.pyfunc.PythonModel*. The *__init__* method will initialize an instance of our *QABot* class and persist it to an class variable.  And a *predict* method will serve as the standard interface for generating a reponse.  That method will receive our inputs as a pandas dataframe but we can write the logic with the knowledge that it will only be receiving one user-provided question at a time:

# COMMAND ----------


# DBTITLE 1,Define MLflow Wrapper for Model
class MLflowQABot(mlflow.pyfunc.PythonModel):
    def __init__(self, qa_chain):
        self.qa_chain = qa_chain

    def predict(self, context, inputs):
        questions = list(inputs["human_reference"])
        # return answer
        return [
            self.qa_chain.generate([{"context": q}]).generations[0][0].text
            for q in questions
        ]


# COMMAND ----------

# MAGIC %md We can then instantiate our model and log the results to compare different model performance

# COMMAND ----------

qbot = MLflowQABot(qa_chain)
test_df = dataset[:5]
qbot.predict([], test_df)

# COMMAND ----------

try:
    spark.read.table("eval_questions")
except:
    print("Creating the evaluation dataset")
    _ = (
        spark.createDataFrame(dataset)
        .write.format("delta")
        .mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable("eval_questions")
    )

# COMMAND ----------

# DBTITLE 1,Persist Evaluation to MLflow
# instantiate mlflow model
model = MLflowQABot(qa_chain)
with mlflow.start_run(run_name=config["model_id"]):
    # Load the dataset and add it to the model run
    questions = mlflow.data.load_delta(table_name="eval_questions")
    mlflow.log_input(questions, context="eval_set")

    # Load model Parameters
    mlflow.log_param("model_id", config["model_id"])

    # mlflow.log_param("prompt template",config['template'])
    mlflow.log_text(config["template"], "prompt.txt")

    # log pipeline_params
    if "pipeline_kwargs" in config:
        mlflow.log_params(config["pipeline_kwargs"])

    if "model_kwargs" in config:
        mlflow.log_params(config["model_kwargs"])

    questions = spark.read.table("eval_questions").toPandas()
    outputs = model.predict(context=None, inputs=questions)

    # Evaluate the model on some example questions
    table_dict = {
        "text": list(questions["human_reference"]),
        "predicted_entities": [output for output in outputs],
        "actual_entities": list(questions["meaning_representation"]),
    }
    mlflow.log_table(table_dict, "eval.json")
    mlflow.end_run()

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
