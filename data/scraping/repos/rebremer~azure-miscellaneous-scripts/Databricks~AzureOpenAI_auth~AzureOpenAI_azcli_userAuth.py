# Databricks notebook source
!pip install pip install azure-cli

# COMMAND ----------

!az login

# COMMAND ----------

!token=$(az account get-access-token --resource=https://cognitiveservices.azure.com/.default --query accessToken --output tsv)

# COMMAND ----------

from azure.identity import DefaultAzureCredential
import openai
import os

access_token = os.getenv('TOKEN')
#print(access_token)

openai.api_base = "https://<<your Azure Open AI instance>>.openai.azure.com/"
openai.api_version = "2023-05-15"
openai.api_type = "azure_ad"
openai.api_key = access_token.token

response = openai.ChatCompletion.create(

    engine="gpt-35-turbo",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Does Azure OpenAI support customer managed keys?"},
        {"role": "assistant", "content": "Yes, customer managed keys are supported by Azure OpenAI."},
        {"role": "user", "content": "Do other Azure AI services support this too?"}
    ]
)

print(response)

# COMMAND ----------
