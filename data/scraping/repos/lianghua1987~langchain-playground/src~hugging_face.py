from dotenv import dotenv_values
from langchain import HuggingFaceHub
import os

print('Welcome to LangChain Playground - Hugging Face')
config = dotenv_values(".env")
os.environ["HUGGINGFACEHUB_API_TOKEN"] = config["HUGGINGFACEHUB_API_TOKEN"]

hf = HuggingFaceHub(repo_id="google/flan-t5-base",
                    model_kwargs={"temperature": 0.5, "max_length": 64},
                    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"])

prompt = """
            What would a good catchy and fancy name be for a github project?
        """

print(hf(prompt))
