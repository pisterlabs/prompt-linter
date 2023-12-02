from dotenv import load_dotenv
from langchain.agents import AgentType, create_csv_agent, initialize_agent
from langchain.agents.agent_toolkits import create_python_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import PythonREPLTool, Tool
from langchain.utilities import SerpAPIWrapper
from langchain.agents import AgentType, initialize_agent
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.tools import PythonREPLTool

# Do this so we can see exactly what's going on under the hood
import langchain

langchain.debug = True


def serp_search(name: str):
    """
    serp search
    """
    serpapi = SerpAPIWrapper()
    result = serpapi.run(f"{name}")
    print("result: ", result)
    return result


load_dotenv()


def main():
    print("Start...")

    # Initialize the OpenAI language model
    python_agent_executor = create_python_agent(
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    search = SerpAPIWrapper()

    search_agent = initialize_agent(
        tools=[
            Tool(
                name="Search",
                func=search.run,
                description="""            
                     Useful when you need to answer questions about current events. 
                     You should ask targeted questions.
                 """,
            )
        ],
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
    )

    grand_agent = initialize_agent(
        tools=[
            Tool(
                name="PythonAgent",
                func=python_agent_executor.run,
                description="""
                        useful when you need to transform natural language and write from it python 
                        and execute the python code,
                        returning the results of the code execution,
                            """,
            ),
            Tool(
                name="Search",
                func=search_agent.run,
                description="""            
                     Useful when you need to answer questions about current events. 
                     You should ask targeted questions.
                 """,
            ),
        ],
        llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
        agent_type=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
    )

    grand_agent.run(
        """
            Tetris Python Code
        """
    )

    # grand_agent.run("print seasons ascending order of the number of episodes they have")


if __name__ == "__main__":
    main()
