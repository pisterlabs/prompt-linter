# Databricks notebook source
# MAGIC %md
# MAGIC # Better Large Language Models (LLMs) With Better Data using Cleanlab Studio
# MAGIC
# MAGIC This notebook (and accompanying [video tutorial](https://www.youtube.com/watch?v=HnC6DwdV4EE)) demonstrates how [Cleanlab Studio](https://cleanlab.ai/) can improve the performance of your LLMs by improving the data they are fine-tuned on, an approach called [Data-centric AI (DCAI)](https://dcai.csail.mit.edu/). You can find this notebook at https://github.com/databricks-industry-solutions/cleanlab-improving-llms.
# MAGIC
# MAGIC In this notebook, we’ll see how Cleanlab Studio systematically improves the training data to boost LLM performance by 37%, without requiring you spending any time or resources to change the model architecture, hyperparameters, or the training process.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC This notebook focuses on _boosting LLM fine-tuning accuracy using Cleanlab Studio_. LLMs acquire powerful generative and discriminative capabilities after being pre-trained on a large corpus of text (usually scraped from the internet), but producing reliable outputs for a particular business use case often requires additional training on a labeled data set from the application domain. This domain-specific training is known as _fine-tuning_ the LLM.
# MAGIC
# MAGIC Labeled data powers AI/ML in the enterprise, but real-world datasets have been found to [contain between 7-50% annotation errors](https://go.cloudfactory.com/hubfs/02-Contents/3-Reports/Crowd-vs-Managed-Team-Hivemind-Study.pdf). Imperfectly-labeled text data hampers the training (and evaluation of) ML models across tasks like intent recognition, entity recognition, and sequence generation. Although pretrained LLMs are equipped with a lot of world knowledge, their performance is adversely affected by noisy training data (as [noted by OpenAI](https://openai.com/research/dall-e-2-pre-training-mitigations)).  This notebook illustrates how using Cleanlab Studio to improve the training data can mitigate the negative effects of bad data (such as erroneous labels) without writing any code or spending any time or resources to change the model architecture, hyperparameters, or training process.
# MAGIC
# MAGIC Because Cleanlab Studio works with data (regardless of which model is used) it remains applicable for LLMs that are yet to be invented, like GPT-10.
# MAGIC
# MAGIC This notebook applies LLMs to a politeness classification task, beginning by fine-tuning OpenAI's Davinci model on the baseline dataset. The model achieves moderate performance on this baseline, but by automatically finding and fixing errors in the data using the [Databricks connector](https://github.com/cleanlab/cleanlab-studio) for [Cleanlab Studio](https://cleanlab.ai/), we can achieve significantly better performance _using the same LLM model and fine-tuning process_, just by improving the data (and spending minimal human time manually reviewing data that is most likely to be erroneous). We see a 37% reduction in prediction error when using Cleanlab Studio to improve the dataset:
# MAGIC
# MAGIC <img src="https://github.com/databricks-industry-solutions/improving-llms-cleanlab/raw/main/images/comparison.png" width="957">
# MAGIC
# MAGIC You would see analogous results whether you are using popular APIs for fine-tuning LLMs or training open-source LLMs like [Dolly](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) or [MPT-7B](https://www.mosaicml.com/blog/mpt-7b) directly [on Databricks](https://www.databricks.com/product/machine-learning/large-language-models).
# MAGIC
# MAGIC See the accompanying blog post for additional context on LLMs and fine-tuning, why data quality matters for LLMs and ML tasks in general, and how [Cleanlab Studio](https://cleanlab.ai/) can help you easily improve ML model robustness and performance by systematically improving data quality.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC This notebook uses Cleanlab Studio to boost LLM accuracy by correcting issues in the data the LLMs are fine-tuned on. Beyond boosting LLM performance, [Cleanlab Studio](https://cleanlab.ai) is an end-to-end platform for both (1) turning unreliable data into reliable insights for business intelligence and analytics teams and (2) training reliable AI solutions on unreliable data for MLOps and technology teams.
# MAGIC
# MAGIC If you don't have a Cleanlab Studio account already, [sign up for an account here](https://cleanlab.ai/). The platform is free to try. It may take up to one day to get access.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Install and configure dependencies
# MAGIC
# MAGIC This notebook uses the fine-tuning APIs [offered by OpenAI](https://platform.openai.com/docs/guides/fine-tuning).
# MAGIC
# MAGIC You can also fine tune open-source LLMs like [Dolly](https://www.databricks.com/blog/2023/04/12/dolly-first-open-commercially-viable-instruction-tuned-llm) or [MPT-7B](https://www.mosaicml.com/blog/mpt-7b) directly [on the Databricks platform](https://www.databricks.com/product/machine-learning/large-language-models).

# COMMAND ----------

!pip install openai

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configure OpenAI API key
# MAGIC
# MAGIC Note that invoking the OpenAI API will use credits or bill you. The estimated cost to run this notebook is $15 with the Davinci model, which is the most powerful but also the most expensive. You can also scale down to the Curie or Ada model to reduce the cost, by setting `openai_model` in the cell below, replacing "davinci" with "curie" or "ada". Fine-tuning on the Ada model costs about $1 per run with the given dataset.
# MAGIC
# MAGIC Put your OpenAI API key in the cell below. You can find your API key at https://platform.openai.com/account/api-keys. Here we have saved the key in a secret scope - see the `RUNME` notebook in this repository for helper scripts to set up the secret scope.

# COMMAND ----------

import openai
import os

# we set the environment variable because it is used by the OpenAI command-line tool
os.environ['OPENAI_API_KEY'] = dbutils.secrets.get("solution-accelerator-cicd","openai_api")  # See the RUNME notebook to setup your OpenAI API key in a secret scope
# we also set the .api_key property below for the Python API
openai.api_key = os.environ['OPENAI_API_KEY']
# set openai model name
openai_model = 'davinci'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download and prepare data
# MAGIC
# MAGIC Here we consider a 3-class variant of the Stanford Politeness Dataset, which has text phrases labeled as: impolite, neutral, or polite. Annotated by human raters, some of these labels are naturally low-quality. 
# MAGIC
# MAGIC The training dataset has 1916 examples each labeled by a single human annotator, and thus some may be unreliable.
# MAGIC
# MAGIC The test dataset has 480 examples each labeled by 5 annotators, and we use their consensus label as a high-quality approximation of the true politeness (measuring test accuracy against these consensus labels). To ensure a fair comparison, this test dataset remains fixed throughout the experiments in this notebook (all data quality improvement is done only for the training set).
# MAGIC
# MAGIC To prepare the data, we download raw data into [DBFS](https://docs.databricks.com/dbfs/index.html), load it into PySpark DataFrames, and do some processing to prepare the dataset for the downstream task.

# COMMAND ----------

# MAGIC %sh
# MAGIC
# MAGIC rm -rf /tmp/stanford-politeness
# MAGIC mkdir -p /tmp/stanford-politeness
# MAGIC cd /tmp/stanford-politeness
# MAGIC curl --silent -L https://raw.githubusercontent.com/databricks-industry-solutions/cleanlab-improving-llms/main/data/train.csv -o train.csv
# MAGIC curl --silent -L https://raw.githubusercontent.com/databricks-industry-solutions/cleanlab-improving-llms/main/data/test.csv -o test.csv
# MAGIC
# MAGIC # move the dataset to our main bucket
# MAGIC rm -rf /dbfs/solacc/product/llm/stanford-politeness/raw
# MAGIC mkdir -p /dbfs/solacc/product/llm/stanford-politeness/raw
# MAGIC cp train.csv test.csv /dbfs/solacc/product/llm/stanford-politeness/raw

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC We can use the `%fs` command to see that our raw data is indeed saved in DBFS.

# COMMAND ----------

# MAGIC %fs ls /solacc/product/llm/stanford-politeness/raw

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we load the raw data into a PySpark DataFrame to enable further processing.

# COMMAND ----------

data_path = '/solacc/product/llm/stanford-politeness'
raw_path = f'{data_path}/raw'
politeness_train_raw = spark.read.options(header='true', inferSchema='true', escape='"', multiLine=True).csv(f'{raw_path}/train.csv')
politeness_test_raw = spark.read.options(header='true', inferSchema='true', escape='"', multiLine=True).csv(f'{raw_path}/test.csv')

# COMMAND ----------

# MAGIC %md
# MAGIC This dataset is missing an index column, but downstream processing that we plan to do requires a unique ID per row. Here, we add monotonically-increasing integer IDs to the rows.

# COMMAND ----------

from pyspark.sql import functions as F

def with_id_column(df):
    df = df.select(F.monotonically_increasing_id().alias("id"), "*")
    return df

politeness_train = with_id_column(politeness_train_raw)
politeness_test = with_id_column(politeness_test_raw)

# COMMAND ----------

# MAGIC %md
# MAGIC We can inspect this prepared data, looking at some specific rows to highlight data errors that are present. For example, the data point with ID `1426` is erroneously labeled "impolite" when "polite" is a more appropriate label.

# COMMAND ----------

display(politeness_train.where((politeness_train.id == 1426) | (politeness_train.id == 299) | (politeness_train.id == 134)))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Formatting data for fine-tuning
# MAGIC
# MAGIC We are using the OpenAI APIs for fine-tuning, which require data in a specific format (JSONL, newline-delimited JSON objects). We also need to do some pre-processing of the label column, adding whitespace before the completion, as the API recommends.
# MAGIC
# MAGIC We save the prepared results into DBFS, so that the result files can be used by the OpenAI API.

# COMMAND ----------

def prepare_data(df, path):
    '''
    Write a dataframe into a single JSONL file located at path, in a format appropriate for fine tuning.

    This makes a small tweak to the data, namely, adding whitespace before the completion.

    We don't need the full power of Spark's parallel and distributed processing for this small demo dataset, but you would want to leverage it for any large real-world datasets. Our small dataset lives in a single partition, but larger datasets would have multiple partitions. By default, Spark writes the dataset into multiple files for efficiency (each partition corresponds to a separate file), but we need a single file to pass to the OpenAI command-line tool. This function ensures that a single JSONL file containing all of the data is produced as the final result.
    '''
    # add whitespace to the completion, as OpenAI requires
    df = df.withColumn('completion', F.format_string(' %s', 'completion'))
    # we don't need the index column here
    df = df.drop('id')
    temp_dir = f'{path}_tmp'
    # write using a single partition, so we have a single JSONL file
    df.coalesce(1).write.mode('overwrite').json(temp_dir)
    # Spark saves the JSON file in a directory, along with some other files we don't need anymore
    all_files = dbutils.fs.ls(temp_dir)
    # move the .json file to the output destination
    for f in all_files:
        if f.path.endswith('.json'):
            dbutils.fs.mv(f.path, path)
    # remove all the other files, which we don't need
    dbutils.fs.rm(temp_dir, recurse=True)

# COMMAND ----------

prepare_data(politeness_train, f'{data_path}/processed/train.jsonl')
prepare_data(politeness_test, f'{data_path}/processed/test.jsonl')

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fine-Tune and evaluate OpenAI model without Cleanlab Studio (accuracy ~65%)
# MAGIC
# MAGIC We use the [OpenAI fine-tuning API](https://platform.openai.com/docs/guides/fine-tuning) to first establish a baseline by:
# MAGIC
# MAGIC - Fine-tuning the OpenAI model on our (original, with some errors) training set
# MAGIC - Evaluating the model on our test set

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC First, we upload our training set and test set to OpenAI:

# COMMAND ----------

train_file = openai.File.create(file=open(f'/dbfs/{data_path}/processed/train.jsonl', 'rb'), purpose='fine-tune')
test_file = openai.File.create(file=open(f'/dbfs/{data_path}/processed/test.jsonl', 'rb'), purpose='fine-tune')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Then, we invoke the fine-tuning API to fine tune the `openai_model` ("davinci" by default, unless you changed it above). Note that this incurs some [cost](https://openai.com/pricing), roughly $7.50 for the Davinci model or $0.50 for the Ada model, at the time this notebook was written.
# MAGIC
# MAGIC We also invoke the fine-tuning API with the `validation_file` keyword argument, so that the API will automatically compute statistics including model accuracy on the test set after the fine-tuning process.

# COMMAND ----------

response = openai.FineTune.create(
    training_file=train_file.id,
    validation_file=test_file.id,
    compute_classification_metrics=True,
    classification_n_classes=3,
    model=openai_model,
    suffix='baseline'
)

# COMMAND ----------

# MAGIC %md
# MAGIC You can follow the progress of fine-tuning with the following command. Once it's done, it'll print "Job complete!". You might need to re-run the cell if it times out. Training time varies based on queue length and other factors; **it can take up to 1 hour to fine-tune the LLM**. The block below would check the status of the finetune and block execution until the finetune is complete. The block is based on this [openai-cookbook example](https://github.com/openai/openai-cookbook/blob/594fc6c952425810e9ea5bd1a275c8ca5f32e8f9/examples/azure/finetuning.ipynb#L278).

# COMMAND ----------

import time

def wait_for_finetune(job_id):
  status = openai.FineTune.retrieve(id=job_id)["status"]
  if status not in ["succeeded", "failed"]:
      print(f'Job not in terminal status: {status}. Waiting.')
      while status not in ["succeeded", "failed"]:
          time.sleep(60)
          status = openai.FineTune.retrieve(id=job_id)["status"]
          print(f'Status: {status}')
  else:
      print(f'Finetune job {job_id} finished with status: {status}')
      
wait_for_finetune(response.id)

# COMMAND ----------

# MAGIC %md
# MAGIC Once the job completes, we see the test accuracy achieved when fine-tuning this LLM on the original training dataset.

# COMMAND ----------

import pandas as pd

!openai api fine_tunes.results -i {response.id} > baseline.csv

base_df = pd.read_csv('baseline.csv')
baseline_acc = base_df.iloc[-1]['classification/accuracy']
print(f"Fine-tuning Accuracy: {baseline_acc:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### Baseline results: ~65% accuracy
# MAGIC
# MAGIC Our baseline Davinci LLM achieves a **test accuracy of 65%** when fine-tuned on the original training data (Curie achieved 64% accuracy, Ada achieved 60% accuracy). Model training is nondeterministic, so your results might vary slightly, even with the exact same dataset and initial model checkpoint. OpenAI's models might also be changed/updated over time.
# MAGIC
# MAGIC Even a state-of-the-art LLM like the Davinci model produces lackluster results for this classification task; is it because of low data quality? 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Improve the data using Cleanlab Studio and re-train the LLM (accuracy ~78%)
# MAGIC
# MAGIC Next, we use the [Databricks connector](https://github.com/cleanlab/cleanlab-studio) for [Cleanlab Studio](https://app.cleanlab.ai/) to automatically improve the data quality, and then re-train our LLM.

# COMMAND ----------

!pip install --upgrade cleanlab-studio
import cleanlab_studio

# COMMAND ----------

# MAGIC %md
# MAGIC ### Set up Cleanlab Studio
# MAGIC
# MAGIC 1. If you don't have an account already, [sign up for an account](https://app.cleanlab.ai/). It may take up to one day to get access.
# MAGIC 2. Get your [API key](https://app.cleanlab.ai/account?tab=General) and enter it below

# COMMAND ----------

CLEANLAB_STUDIO_API_KEY = dbutils.secrets.get("solution-accelerator-cicd","cleanlab_api")  # See the RUNME notebook to setup your Cleanlab Studio API key in a secret scope
studio = cleanlab_studio.Studio(CLEANLAB_STUDIO_API_KEY)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Upload dataset to Cleanlab Studio
# MAGIC Next, we can directly upload a Spark DataFrame to Cleanlab Studio by passing it to `studio.upload_dataset()`.

# COMMAND ----------

dataset_id = studio.upload_dataset(politeness_train, dataset_name='Stanford Politeness')

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create a project
# MAGIC
# MAGIC To analyze the data, use the [Cleanlab Studio web UI](https://app.cleanlab.ai/) to create a project, configuring it according to the ML task. For this demo, you should select:
# MAGIC
# MAGIC - ML task: text classification
# MAGIC - Type of classification: multi-class
# MAGIC - Text column: prompt (will be auto-detected)
# MAGIC - Label column: completion (will be auto-detected)
# MAGIC
# MAGIC Select fast mode or regular mode depending on the speed/quality tradeoff you desire.
# MAGIC
# MAGIC <img src="https://github.com/databricks-industry-solutions/improving-llms-cleanlab/raw/main/images/create-project.png" width="1440">

# COMMAND ----------

# MAGIC %md
# MAGIC ### Make corrections to your dataset with Cleanlab Studio
# MAGIC
# MAGIC Cleanlab Studio not only finds data points with potential issues, but it also makes suggestions for how to address the issues (e.g., changing the label of a data point). Deciding how to make use of the analysis results is up to you. For example, you could discard all potentially erroneous data points, or you could review the data points most likely to have issues and make corrections. This human-in-the-loop data correction usually gives the best results.
# MAGIC
# MAGIC If you want to save time, you could briefly review some flagged issues, and then auto-fix the top issues.
# MAGIC
# MAGIC <img src="https://github.com/databricks-industry-solutions/improving-llms-cleanlab/raw/main/images/make-corrections.png" width="1523">

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC The selected row in the screenshot above is an example of a poorly-labeled datapoint. The phrase:
# MAGIC
# MAGIC > I’ll take a look at getLogEntries when I have time. Would you mind adding me as a committer?
# MAGIC
# MAGIC is labeled "impolite". Cleanlab Studio flags this as a label error, and it suggests that the label be switched to "polite". In the screenshot above, we pressed "W" to accept Cleanlab Studio's suggestion to automatically fix the label.
# MAGIC
# MAGIC Label issues like this cause the accuracy of the fine-tuned LLM to be degraded. Correcting these issues allows us to train an improved LLM, as we'll see below.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Export your improved dataset back into Databricks
# MAGIC
# MAGIC Once you're done correcting issues found in your dataset with Cleanlab Studio, export the improved dataset back into Databricks:
# MAGIC
# MAGIC 1. Click on the "Export Cleanset" button in your Cleanlab Studio project
# MAGIC 2. Select the "Export using API" tab
# MAGIC 3. Copy the "cleanset ID" and paste it into the cell below

# COMMAND ----------

cleanset_id = '7b27d51ba79b4087b32b3f064f87a47b'  # paste your own Cleanset ID here
politeness_train_fixed = studio.apply_corrections(cleanset_id, politeness_train)
display(politeness_train_fixed)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fine-tune the LLM on your improved dataset and evaluate the results
# MAGIC
# MAGIC Let's see how Cleanlab Studio improves the performance of the LLM. We follow the same process as earlier, except we use the `politeness_train_fixed` DataFrame as our training data.
# MAGIC
# MAGIC When we ran the experiment below, we used Cleanlab Studio's web interface to review the data issues that it flagged. Machine-augmented human-in-the-loop data improvement often gives the best results. If you want to use the dataset that we exported from Cleanlab Studio, uncomment the line below.

# COMMAND ----------

# By default for reproducibility, we use the dataset that we exported from Cleanlab Studio as csv
# But if you want to use your dataset that you improved, downloaded as politeness_train_fixed above
# set the flag below to 'False'
use_provided_training_set_improved_using_cleanlab_studio = True
if use_provided_training_set_improved_using_cleanlab_studio:
    politeness_train_fixed = pd.read_csv('https://raw.githubusercontent.com/databricks-industry-solutions/cleanlab-improving-llms/main/data/train_fixed.csv')
    politeness_train_fixed = with_id_column(spark.createDataFrame(politeness_train_fixed))

# COMMAND ----------

prepare_data(politeness_train_fixed, f'{data_path}/processed/train_fixed.jsonl')

# COMMAND ----------

train_file_fixed = openai.File.create(file=open(f'/dbfs/{data_path}/processed/train_fixed.jsonl', 'rb'), purpose='fine-tune')

# COMMAND ----------

response_fixed = openai.FineTune.create(
    training_file=train_file_fixed.id,
    validation_file=test_file.id,
    compute_classification_metrics=True,
    classification_n_classes=3,
    model=openai_model,
    suffix='fixed'
)

# COMMAND ----------

# MAGIC %md
# MAGIC You can follow the progress of fine-tuning with the following command. Once it's done, it'll print "Job complete!". You might need to re-run the cell if it times out. Training time varies based on queue length and other factors; it can take up to 1 hour to fine-tune the LLM. We use the `wait_for_finetune` function defined before to block this step until the finetuning is done.

# COMMAND ----------

wait_for_finetune(response_fixed.id)

# COMMAND ----------

# MAGIC %md
# MAGIC Once the job completes, we see the test accuracy achieved when fine-tuning this LLM on the improved dataset. If you simply auto-fixed some of the labels (spending zero human time on data improvement), you'll still see improvement; if you reviewed some of Cleanlab Studio's suggestions following a human-in-the-loop data cleaning process, you'll see larger improvements here.

# COMMAND ----------

!openai api fine_tunes.results -i {response_fixed.id} > fixed.csv

fixed_df = pd.read_csv('fixed.csv')
fixed_acc = fixed_df.iloc[-1]['classification/accuracy']
print(f"Fine-tuning Accuracy: {fixed_acc:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Impact of data improvement using Cleanlab Studio: ~78% accuracy (compared to ~65% baseline accuracy)
# MAGIC
# MAGIC Training on the improved dataset, we see a **test accuracy of 78%** for the Davinci model (Curie achieved ~76% accuracy, Ada achieved ~75% accuracy). These results are from our `train_fixed.csv` (provided above); results on your dataset will vary depending on how you improved the dataset using Cleanlab Studio (e.g., whether you used auto-fix or manually reviewed the top issues, how you corrected labels, how you removed outliers, etc.). Even the results of fine-tuning on the provided dataset might vary slightly, because model training is nondeterministic, and OpenAI's initial model checkpoints may be updated over time.
# MAGIC
# MAGIC In this evaluation, we see that data quality has a huge impact on LLM performance. **By simply improving the data quality** (and leaving the original LLM checkpoint, training parameters, fine-tuning process, etc. as-is), we have **reduced prediction error by ~37%**.

# COMMAND ----------

# MAGIC %md
# MAGIC # Takeaway: Use Cleanlab Studio to turn unreliable data into more reliable insights and models
# MAGIC
# MAGIC Errors like outliers and label issues are [common in real-world datasets](https://labelerrors.com), and these errors can have a dramatic impact on the reliability and robustness of ML models trained on this data as well as insights and analytics obtained. [Cleanlab Studio](https://cleanlab.ai/) is a solution for dealing with erroneous or noisy data via AI automated techniques to help avoid the tedious manual effort that data scientists often dread. Cleanlab Studio helps you efficiently find and fix data and label issues for any ML model (not just LLMs) and most types of data (not just text, but also images, audio, tabular data, etc.) without requiring you to write code or have any ML expertise. In the case study in this post, we saw how Cleanlab Studio boosted the performance of an LLM fine-tuned for a classification task by 37% without spending any time or resources to change the model architecture, hyperparameters, or the training process.
# MAGIC
# MAGIC Because Cleanlab Studio improves models and insights by improving the underlying data, it works for any model or LLM that exists today or may exist in the future and will only become better at identifying issues as more accurate models are released!
