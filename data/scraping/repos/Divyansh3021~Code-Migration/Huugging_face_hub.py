from langchain import HuggingFaceHub
from getpass import getpass
 
HUGGINGFACEHUB_API_TOKEN = ""
 
import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
 
repo_id = "databricks/dolly-v2-12b"
 
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0, "max_length":64})