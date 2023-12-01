import os

os.environ["OPENAI_API_TYPE"] = "azure"
os.environ["OPENAI_API_VERSION"] = "2023-03-15-preview"
os.environ["OPENAI_API_BASE"] = "https://cast-southcentral-nprd-apim.azure-api.net/api/"
os.environ["OPENAI_API_KEY"] = "bb98f2a8ef604e098ac025ec373d6951"   

from langchain.llms import AzureOpenAI

llmOpenAI = AzureOpenAI( deployment_name="text-davinci-003")

result = llmOpenAI("Provide terraform code to create a resource group in Azure")
print(result)

