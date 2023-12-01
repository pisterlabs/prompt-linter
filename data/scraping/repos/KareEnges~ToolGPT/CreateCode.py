from langchain.chat_models import ChatOpenAI
from langchain.agents import AgentType
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools import StructuredTool
from langchain.tools.python.tool import PythonREPLTool

import config


def CreateCode(prompt):
    """根据prompt的要求生成代码并运行代码"""
    agent_executor = create_python_agent(
        llm=ChatOpenAI(temperature=0, openai_api_key=config.OPENAI_API_KEY, model_name="gpt-3.5-turbo-16k"),
        tool=PythonREPLTool(),
        verbose=True,
        agent_type=AgentType.OPENAI_FUNCTIONS,
        agent_executor_kwargs={"handle_parsing_errors": True}
    )
    return agent_executor.run(prompt)


CreateCodeTool = StructuredTool.from_function(CreateCode)
