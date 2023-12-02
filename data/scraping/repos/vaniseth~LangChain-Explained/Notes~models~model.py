from langchain.llms import OpenAI
from langchain import HuggingFaceHub
from langchain.llms import Cohere
import dotenv


from dotenv import load_dotenv
load_dotenv()

llm = OpenAI(model_name = 'test-01')
llm = HuggingFaceHub(repo_id = 'google/flan-t5-xl')
llm = Cohere()

result = llm('Tell me a joke')