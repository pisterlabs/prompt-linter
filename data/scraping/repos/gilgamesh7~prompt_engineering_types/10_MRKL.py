# Modular Reasoning, Knowledge and Language

# SQLDatabaseChain DEPRECATED ?

import os
import openai
import dotenv

from langchain import LLMMathChain, OpenAI, SerpAPIWrapper, SQLDatabase, SQLDatabaseChain
from langchain.agents import initialize_agent, Tool

dotenv.load_dotenv()

openai.api_key=os.environ.get('OPENAI_API_KEY')

llm = OpenAI(temperature=0)
search = SerpAPIWrapper()
llm_math_chain = LLMMathChain(llm=llm, verbose=True)
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
db_chain = SQLDatabaseChain(llm=llm, database=db, verbose=True)
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. You should ask targeted questions"
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    ),
    Tool(
        name="FooBar DB",
        func=db_chain.run,
        description="useful for when you need to answer questions about FooBar. Input should be in the form of a question containing full context"
    )
]
mrkl = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

mrkl.run("Who is Leo DiCaprio's girlfriend? What is her current age raised to the 0.43 power?")
