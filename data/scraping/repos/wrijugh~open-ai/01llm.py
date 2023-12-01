from langchain.llms import AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from dotenv import load_dotenv
import os

if load_dotenv("../.env"):
    print("Found OpenAPI Base Endpoint: " + os.getenv("OPENAI_API_BASE"))
else: 
    print("No file .env found")

openai_api_type = os.getenv("OPENAI_API_TYPE")
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_api_base = os.getenv("OPENAI_API_BASE")
openai_api_version = os.getenv("OPENAI_API_VERSION")
deployment_name = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME")
model_name = os.getenv("OPENAI_COMPLETION_MODEL") 
embedding_name = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")


def generate_pet_name():
    
    llm = AzureOpenAI(
        model_name = model_name,
        deployment_name = deployment_name,
        temperature = 0.5    
    )

    name = llm("I have a light brown dog and I want a cool name for him. Suggest five cool names for my dog.")

    return name

if __name__ == "__main__":
    print(generate_pet_name())