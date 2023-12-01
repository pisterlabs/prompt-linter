import os
import openai
import dotenv

from langchain import OpenAI, Wikipedia
from langchain.agents import initialize_agent, Tool
from langchain.agents.react.base import DocstoreExplorer

dotenv.load_dotenv()

openai.api_key=os.environ.get('OPENAI_API_KEY')

docstore=DocstoreExplorer(Wikipedia())

tools = [
    Tool(
        name="Search",
        func=docstore.search,
        description="Try to search for wiki page."
    ),
    Tool(
        name="Lookup",
        func=docstore.lookup,
        description="Lookup a term in the page, imitating cmd-F functionality"
    )
]

llm = OpenAI(temperature=0, model_name="text-davinci-003")

react = initialize_agent(tools, llm, agent="react-docstore", verbose=True)

question = "What profession does Nicholas Ray and Elia Kazan have in common?"

react.run(question)