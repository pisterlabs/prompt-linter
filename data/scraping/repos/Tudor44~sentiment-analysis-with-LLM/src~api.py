
from dotenv import find_dotenv, load_dotenv
from langchain.llms import OpenAI

class Api:
    
    def __init__(self):
        load_dotenv(find_dotenv())
        self.llm = OpenAI(temperature=0, model_name="gpt-3.5-turbo")
        self.tools = []