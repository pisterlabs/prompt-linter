from langchain import OpenAI, SerpAPIWrapper
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import OpenAI, SerpAPIWrapper, LLMChain
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
import os

# Load environment variables from .env file
load_dotenv()

# Access the API key from the environment variable
serpapi_api_key = os.getenv('SERPAPI_API_KEY')


#embeddings = OpenAIEmbeddings(openai_api_key="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")
#os.environ["OPENAI_API_KEY"] = "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"  # https://platform.openai.com (Thx Michael from Twitter)
#os.environ['SERPAPI_API_KEY'] = 'd4eb38ff06e003ba07a08950ec770d7d3b876e5685ff9806d3a79a5dc339e558' # https://serpapi.com/



llm = ChatOpenAI(temperature=0)
search = SerpAPIWrapper(serpapi_api_key='d4eb38ff06e003ba07a08950ec770d7d3b876e5685ff9806d3a79a5dc339e558')
from langchain import OpenAI, Wikipedia
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.agents.react.base import DocstoreExplorer
docstore=DocstoreExplorer(Wikipedia())
tools = [
    Tool(
        name="Search",
        func=search.run,
        description="useful for when you need to ask with search"
    ),
    
]

#llm = ChatOpenAI(temperature=0)
llm = ChatOpenAI(temperature=0)

react = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

question = "what are the best 3 Github repositories for impedance control using ros2"
react.run(question)