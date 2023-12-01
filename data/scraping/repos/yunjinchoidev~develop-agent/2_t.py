from dotenv import load_dotenv
from langchain.agents import AgentType
from langchain.agents.agent_toolkits import create_python_agent
from langchain.chat_models import ChatOpenAI
from langchain.tools import PythonREPLTool
from langchain.schema import BaseOutputParser
import os
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

# langchain.debug = True

load_dotenv()


def main():

    python_agent_executor = create_python_agent(
        llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
        tool=PythonREPLTool(),
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
    )

    python_agent_executor.run(
        """
        Develop a webpage with a box that can be moved using the mouse.
        Save the HTML file in a static folder named 'index.html'.
        Pay attention to triple-quoted string literals.
        Map it using FastAPI.
        There is no need to install anything.
        Use HTMLResponse to serve your HTML file.
        Save the FastAPI file as 'app.py'.
        The available ports you can use are 4500 and 7000.
        Use uvicorn to launch the FastAPI server, making it directly accessible.
    """
    )


if __name__ == "__main__":
    main()
