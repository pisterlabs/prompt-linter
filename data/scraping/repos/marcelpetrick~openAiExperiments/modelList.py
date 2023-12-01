import os
import openai

openai.organization = "org-7TH0uzFIufmpM9zDARVHjTxk"
openai.api_key = os.getenv("OPENAI_API_KEY")
print(f"model list: {openai.Model.list()}")
