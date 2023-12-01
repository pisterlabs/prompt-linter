from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.llms import OpenAI, AzureOpenAI
from langchain.chat_models import AzureChatOpenAI
from langchain import SerpAPIWrapper
from langchain.python import PythonREPL
from langchain import LLMMathChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import HumanMessage
import openai, os
from patch import log_function_call, patch_langchain

patch_langchain()

openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = os.environ["OPENAI_API_BASE"]
openai.api_key = os.environ["OPENAI_API_KEY"]

# llm = AzureOpenAI(deployment_name="text-davinci-003")
llm = AzureOpenAI(deployment_name="text-davinci-003", model_name="text-davinci-003", temperature=0)

# llm = AzureChatOpenAI(
#     deployment_name="gpt-4",
#     temperature=0,
#     openai_api_version="2023-03-15-preview",
# )
# llm = AzureChatOpenAI(
#     deployment_name="gpt-35-turbo",
#     temperature=0,
#     openai_api_version="2023-03-15-preview",
# )
print(llm)
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
#agent.run("Where do these 2 lines intersect? y = 2x + 1, y = -3x + 2")
agent.run("I have 1000 dollars. given today's governement bond rate, how much money would I have in 2 years if I invested them in 2 year government bonds?")
