import functools
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Create a formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)

indent = 0
def log_function_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        global indent
        indent += 1
        #logger.info(f"Function call: {func.__module__}.{func.__name__}({args}, {kwargs})")
        print(f"\n{'  '*indent}Function call: {func.__module__}.{func.__name__}(")
        for v in args:
            print(f"{'  '*indent}    {v}")
        for k, v in kwargs.items():
            print(f"{'  '*indent}    {k}: {v}")
        print(")")
        result = func(*args, **kwargs)
        #logger.info(f"Function result: {result}")
        print(f"{'  '*indent}Function result: {result}")
        indent -= 1
        return result
    return wrapper

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

AzureOpenAI._generate = log_function_call(AzureOpenAI._generate)
AzureChatOpenAI._generate = log_function_call(AzureChatOpenAI._generate)
SerpAPIWrapper.run = log_function_call(SerpAPIWrapper.run)
PythonREPL.run = log_function_call(PythonREPL.run)
LLMMathChain.run = log_function_call(LLMMathChain.run)

openai.api_type = "azure"
openai.api_version = "2022-12-01"
openai.api_base = "https://aoai.openai.azure.com/"
openai.api_key = os.environ["OPENAI_API_KEY"]

# llm = AzureChatOpenAI(
#     deployment_name="gpt-4",
#     temperature=0,
#     openai_api_version="2023-03-15-preview",
# )
llm = AzureOpenAI(deployment_name="text-davinci-003", model_name="text-davinci-003")
tools = load_tools(["serpapi", "llm-math"], llm=llm)
agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
#agent.run("Where do these 2 lines intersect? y = 2x + 1, y = -3x + 2")
agent.run("I have 1000 dollars. given today's governement bond rate, how much money would I have in 2 years if I invested them in 2 year government bonds?")

