import openai
import pandas as pd
import json
import time
import os
import sys
import chromadb
import openai
import requests
import pandas as pd
import numpy as np
import config
from langchain.docstore.document import Document
from langchain.document_loaders import CSVLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import CSVLoader
from langchain.document_loaders import DirectoryLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.agents.agent_types import AgentType
from langchain.agents import create_csv_agent
from typing import List
from langchain.docstore.document import Document
from langchain.retrievers import ChatGPTPluginRetriever
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score



openai.api_key = config.api_key
original_data = pd.read_csv("train.csv")

resampled_data_positive = original_data[original_data["STATUS"] == 1]
resampled_data_negative = original_data[original_data["STATUS"] == 2]

if len(resampled_data_positive) > len(resampled_data_negative):
    resampled_data = pd.concat([resampled_data_negative, resampled_data_positive[:len(resampled_data_negative)]], ignore_index=True).sample(frac =1, random_state=42)
else:
    resampled_data = pd.concat([resampled_data_positive, resampled_data_negative[:len(resampled_data_positive)]], ignore_index=True).sample(frac =1, random_state=42)


print(len(resampled_data_negative), len(resampled_data_positive), len(resampled_data))
print("-" * 150)

train_df, test_val_df = train_test_split(resampled_data, test_size=0.4, random_state=42)
validate_df, test_df = train_test_split(test_val_df, test_size=0.5, random_state=42)

# Save the split data into separate CSV files
train_df.to_csv('train_data.csv', index=False)
validate_df.to_csv('validate_data.csv', index=False)
test_df.to_csv('test_data.csv', index=False)

r_train_data = []
r_test_data = []
r_validate_data = []

def train_data_write_jsonl(path: str, documents: List[Document]) -> None:
    lookup = {"2": "negative", "1": "positive"}
    for d in documents:
        data_dict = {}
        for k in d.page_content.split("\n"):
            if k.split(": ")[0] != "STATUS":
                key, value = k.split(": ")
                data_dict[key] = value
        r_train_data.append([str(data_dict), lookup[d.metadata["source"]]])
    # print(r_train_data[0])
    messages = []
    for line in r_train_data:
        msg = {"messages": [{"role": "system", "content": "You are a very helpful assistant that classifies liver diseases"}, {"role": "user", "content":line[0]}, {"role": "assistant", "content": str(line[1])}]}
        messages.append(msg)
    with open(path, "w") as f:
        for line in messages:
            json.dump(line, f)
            f.write("\n")


def validate_data_write_jsonl(path: str, documents: List[Document]) -> None:
    lookup = {"2": "negative", "1": "positive"}
    for d in documents:
        data_dict = {}
        for k in d.page_content.split("\n"):
            if k.split(": ")[0] != "STATUS":
                key, value = k.split(": ")
                data_dict[key] = value
        r_validate_data.append([str(data_dict), lookup[d.metadata["source"]]])
    # print(r_train_data[0])
    messages = []
    for line in r_validate_data:
        msg = {"messages": [{"role": "system", "content": "You are a very helpful assistant that classifies liver diseases"}, {"role": "user", "content":line[0]}, {"role": "assistant", "content": str(line[1])}]}
        messages.append(msg)
    with open(path, "w") as f:
        for line in messages:
            json.dump(line, f)
            f.write("\n")


def test_data_write_jsonl(path: str, documents: List[Document]) -> None:
    lookup = {"2": "negative", "1": "positive"}
    for d in documents:
        data_dict = {}
        for k in d.page_content.split("\n"):
            if k.split(": ")[0] != "STATUS":
                key, value = k.split(": ")
                data_dict[key] = value
        r_test_data.append([str(data_dict)])
    # print(r_train_data[0])
    messages = []
    for line in r_test_data:
        msg = {"messages": [{"role": "system", "content": "You are a very helpful assistant that classifies liver diseases"}, {"role": "user", "content": "Please classify the following patient data as Positive or negative based on the provided training data:"}, {"role": "user", "content": str(line[0])}]}
        messages.append(msg)
    with open(path, "w") as f:
        for line in messages:
            json.dump(line, f)
            f.write("\n")



train_loader = CSVLoader("./train_data.csv", source_column="STATUS")
validate_loader = CSVLoader("./validate_data.csv", source_column="STATUS")
test_loader = CSVLoader("./test_data.csv", source_column="STATUS")


train_data_write_jsonl("foo_train.jsonl", train_loader.load())
validate_data_write_jsonl("foo_validate.jsonl", validate_loader.load())
test_data_write_jsonl("foo_test.jsonl", test_loader.load())



status_good = ["uploaded", "processed", "pending", "succeeded"]
status_bad = ["error", "deleting", "deleted"]

all_uploded_files = openai.File.list()["data"]

for file in all_uploded_files:
    try :
        openai.File.delete(file["id"])
    except Exception as e:
        print(f"Failed to delete file {file}: {e}")


training_file = openai.File.create(
  file=open("foo_train.jsonl", "r"),
  user_provided_filename="foo_train.jsonl",
  purpose='fine-tune'
)

while training_file.status not in status_good:
    print(training_file.status)
print("Train File processed")
print("-" * 150)

validation_file = openai.File.create(
    file=open("foo_validate.jsonl", "r"),
    user_provided_filename="foo_validate.jsonl",
    purpose='fine-tune'
)

while validation_file.status not in status_good:
    print(validation_file.status)
print("Validation File processed")
print("-" * 150)

# test_file = openai.File.create(
#     file=open("foo_test.jsonl", "r"),
#     user_provided_filename="foo_test.jsonl",
#     purpose='fine-tune'
# )

# while test_file.status not in status_good:
#     print(test_file.status)
# print("Test File processed")
# print("-" * 150)

current_exsiting_files = openai.File.list()["data"]
print(current_exsiting_files)
print("-" * 150)

fine_tuned_model_list = openai.FineTuningJob.list()
fine_tuned_models_id_list = []

for model in fine_tuned_model_list["data"]:
    if model["fine_tuned_model"] != None:
        fine_tuned_models_id_list.append({"created_at" : model["created_at"],"finished_at" : model["finished_at"],"model_id" : model["fine_tuned_model"]})
# print(fine_tuned_models_id_list)

all_available_models = openai.Model.list()["data"]
my_org_models_ids = []

for model in all_available_models:
    if model["permission"][0]["organization"] == "org-T235lb7vYT32FxvDLk6ASY6e" and model["id"] != "ft:gpt-3.5-turbo-0613:personal:ldc-final:8GCaBmDY":
        my_org_models_ids.append(model["id"])
print("my_org_models_ids_before_deleting", my_org_models_ids)
print("-" * 150)

for org_model in my_org_models_ids:
    try : 
        openai.Model.delete(org_model)
    except Exception as e:
        print(f"Failed to delete model {org_model}: {e}")

after_deleting_models_list = [model for model in openai.Model.list()["data"] if model["permission"][0]["organization"] == "org-T235lb7vYT32FxvDLk6ASY6e"]
print("after_deleting_models_list", after_deleting_models_list)
print("-" * 150)


if openai.Model.retrieve("ft:gpt-3.5-turbo-0613:personal:ldc-final:8GCaBmDY") is not None:
    fine_tune_job = openai.FineTuningJob.create(
        model = "gpt-3.5-turbo-0613",
        training_file = training_file.id,
        validation_file = validation_file.id,
        suffix = "LDC-final",
        hyperparameters={"n_epochs":10},
    )
    while fine_tune_job.status not in status_good:
        if fine_tune_job.status == "failed":
            print(fine_tune_job.status)
            print(fine_tune_job["error"])
            break
        else:
            print(fine_tune_job.status)
else:
    print("Model already exists")
    print(openai.Model.retrieve("ft:gpt-3.5-turbo-0613:personal:ldc-final:8GCaBmDY"))
    print("-" * 150)

predicted_output = []
with open("foo_test.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        messages = data["messages"]
        response = openai.ChatCompletion.create(
            model="ft:gpt-3.5-turbo-0613:personal:ldc-final:8GCaBmDY",
            messages=messages,
            
        )
        predicted_output.append(response)

AI_fine_tuned_pred = []
for response in predicted_output :
    if response["choices"][0]["message"]["content"] == "positive" or response["choices"][0]["message"]["content"] == "Positive":
        AI_fine_tuned_pred.append(1)
    elif response["choices"][0]["message"]["content"] == "negative" or response["choices"][0]["message"]["content"] == "Negative":
        AI_fine_tuned_pred.append(2)
    else:
        AI_fine_tuned_pred.append(response["choices"][0]["message"]["content"])

print(test_df["STATUS"].values, len(test_df["STATUS"].values))
print("-" * 150)

print(AI_fine_tuned_pred, len(AI_fine_tuned_pred), len(predicted_output))
print("-" * 150)
AI_model_accuracy = accuracy_score(test_df["STATUS"].values, AI_fine_tuned_pred)

print("AI_model_accuracy", AI_model_accuracy)
print("-" * 150)

AI_model_f1_score = f1_score(test_df["STATUS"].values, AI_fine_tuned_pred)

print("AI_model_f1_score", AI_model_f1_score)
print("-" * 150)

AI_model_confusion_matrix = confusion_matrix(test_df["STATUS"].values, AI_fine_tuned_pred)

print("AI_model_confusion_matrix", AI_model_confusion_matrix)
print("-" * 150)

AI_model_classification_report = classification_report(test_df["STATUS"].values, AI_fine_tuned_pred)
print(AI_model_classification_report)
print("-" * 150)





