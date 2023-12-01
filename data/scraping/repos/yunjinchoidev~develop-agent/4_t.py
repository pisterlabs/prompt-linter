from langchain.agents.agent_toolkits import create_python_agent
from langchain.utilities import SerpAPIWrapper
from langchain.agents import AgentType, initialize_agent
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool
from langchain.tools import PythonREPLTool

load_dotenv()


def find_code_search_by_serp(name: str):

    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")

    search = SerpAPIWrapper()

    # Define a list of tools offered by the agent
    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="""            
                Useful when you need to answer questions about current events. 
                You should ask targeted questions.
            """,
        ),
        Tool(
            name="python_repl",
            func=PythonREPLTool(),
            description="python_repl, useful for when you need to run python code, and get the output, or save the output to a file",
        ),
    ]

    mrkl = initialize_agent(
        tools=tools, llm=llm, agent=AgentType.OPENAI_MULTI_FUNCTIONS, verbose=True
    )

    run = mrkl.run(name)

    with open("result.txt", "w") as f:
        f.write(run)

    return run


if __name__ == "__main__":

    x = find_code_search_by_serp(
        "Neumorphism style Todo web page that can add, delete, control todo list. The ports you can use are 4500 and 7000."
    )

    python_agent_executor = create_python_agent(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    python_agent_executor.run(
        """
        I provide a todo web page that can add, delete, control todo list.
        Follow this and Develope
        The ports you can use are 4500 and 7000.
        You Should Run Server.
        : 
        """
        + x
    )

    # main()
