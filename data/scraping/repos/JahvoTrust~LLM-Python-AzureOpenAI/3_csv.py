import os
import openai
from langchain.llms import AzureOpenAI
from dotenv import load_dotenv
from langchain import OpenAI 
from langchain.document_loaders.csv_loader import CSVLoader

load_dotenv()

# Configure OpenAI API
openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = os.getenv('OPENAI_API_BASE')
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create a completion - qna를 위하여 davinci 모델생성
llm = AzureOpenAI(deployment_name="text-davinci-003")

filepath = "db/csvdata.csv"
loader = CSVLoader(filepath)
data = loader.load()
print(data)

# llm = OpenAI(temperature=0)

from langchain.agents import create_csv_agent
agent = create_csv_agent(llm, filepath, verbose=True)
# agent.run("who can behave in a predictable way?")
agent.run("what is YC?")
# agent.run("List the top 3 devices that the respondents use to submit their responses")
# agent.run("Consider iOS and Android as mobile devices. What is the percentage of respondents that discovered us through social media submitting this from a mobile device?")
