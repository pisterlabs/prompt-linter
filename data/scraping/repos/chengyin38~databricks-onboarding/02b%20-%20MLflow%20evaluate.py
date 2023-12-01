# Databricks notebook source
# MAGIC %pip install "mlflow-skinny[databricks]>=2.4.1" "rouge_score==0.1.2"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import pandas as pd

import mlflow

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

os.environ["OPENAI_API_KEY"] = dbutils.secrets.get("llm_scope", "openai_token") # This is personal token 

assert (
    "OPENAI_API_KEY" in os.environ
), "Please set the OPENAI_API_KEY environment variable to run this example."

# COMMAND ----------

def build_and_evalute_model_with_prompt(prompt_template):
    with mlflow.start_run() as run:
      mlflow.log_param("prompt_template", prompt_template)
      # Create a news summarization model using prompt engineering with LangChain. Log the model
      # to MLflow Tracking
      llm = OpenAI(temperature=0.1)
      prompt = PromptTemplate(input_variables=["article"], template=prompt_template)
      chain = LLMChain(llm=llm, prompt=prompt)
      logged_model = mlflow.langchain.log_model(chain, artifact_path="model")

      # Evaluate the model on a small sample dataset
      sample_data = spark.table("chengyin.summarization_example_data").toPandas() #pd.read_csv("summarization_example_data.csv")
      mlflow.evaluate(
          model=logged_model.model_uri,
          model_type="text-summarization",
          data=sample_data,
          targets="highlights",
      )
      return run

# COMMAND ----------

prompt_template_1 = (
    "Write a summary of the following article that is between triple backticks: ```{article}```"
)
print(f"Bulding and evaluating model with prompt: '{prompt_template_1}'")
run1 = build_and_evalute_model_with_prompt(prompt_template_1)

# COMMAND ----------

prompt_template_2 = (
    "Write a summary of the following article that is between triple backticks. Be concise. Make"
    " sure the summary includes important nouns and dates and keywords in the original text."
    " Just return the summary. Do not include any text other than the summary: ```{article}```"
)
print(f"Building and evaluating model with prompt: '{prompt_template_2}'")
run2 = build_and_evalute_model_with_prompt(prompt_template_2)

# COMMAND ----------

# Load the evaluation results
results: pd.DataFrame = mlflow.load_table(
    "eval_results_table.json", extra_columns=["run_id", "params.prompt_template"]
)
results_grouped_by_article = results.sort_values(by="id")
print("Evaluation results:")
display(results_grouped_by_article[["run_id", "params.prompt_template", "article", "outputs"]])

# COMMAND ----------

# Score the best model on a new article
new_article = """
Adnan Januzaj swapped the lush turf of Old Trafford for the green baize at Sheffield when he
turned up at the snooker World Championships on Wednesday. The Manchester United winger, who has
endured a frustrating season under Louis van Gaal, had turned out for the Under 21 side at Fulham
on Tuesday night amid reports he could be farmed out on loan next season. But Januzaj may want to
consider trying his hand at another sport after displaying his silky skillls on a mini pool table.
Adnan Januzaj (left) cheered on\xa0Shaun Murphy (right) at the World Championship in Sheffield.
Januzaj shows off his potting skills on a mini pool table at the Crucible on Wednesday.
The 20-year-old Belgium international was at the Crucible to cheer on his friend Shaun Murphy in
his quarter-final against Anthony McGill. The 2005 winner moved a step closer to an elusive second
title in Sheffield with a 13-8 victory, sealed with a 67 break. Three centuries in the match, and
the way he accelerated away from 6-6, showed Murphy is a man to fear, and next for him will be
Neil Robertson or Barry Hawkins. Januzaj turned out for Under 21s in the 4-1 victory at Fulham on
Tuesday night.
"""

print(
    f"Scoring the model with prompt '{prompt_template_2}' on the article '{new_article[:70] + '...'}'"
)

# COMMAND ----------

best_model = mlflow.pyfunc.load_model(f"runs:/{run2.info.run_id}/model")
summary = best_model.predict({"article": new_article})
print(f"Summary: {summary}")

# COMMAND ----------


