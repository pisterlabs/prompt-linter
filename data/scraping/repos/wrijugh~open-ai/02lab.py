import openai

import os
from dotenv import load_dotenv

# Load environment variables
if load_dotenv("../.env"):
    print("Found OpenAI API Base Endpoint: " + os.getenv("OPENAI_API_BASE"))
else: 
    print("OpenAI API Base Endpoint not found. Have you configured the .env file?")

openai.api_type = os.getenv("OPENAI_API_TYPE")
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base = os.getenv("OPENAI_API_BASE")
# This version of the API is needed to properly retrieve the list of model deployments.
openai.api_version = "2023-03-15-preview"

openai.Deployment.list()

# Grab API Version in the .env file.
openai.api_version = os.getenv("OPENAI_API_VERSION")
COMPLETION_MODEL = os.getenv("OPENAI_COMPLETION_MODEL")
DEPLOYMENT_ID = os.getenv("AZURE_OPENAI_COMPLETION_DEPLOYMENT_NAME")

r = openai.ChatCompletion.create(
    model = COMPLETION_MODEL,
    deployment_id = DEPLOYMENT_ID,
    messages = [{"role" : "assistant", "content" : "The one thing I love more than anything else is "}],
)

print(r)

print(r.choices[0].message.content)