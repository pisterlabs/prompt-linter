from langchain.tools import BaseTool
from langchain.agents import Tool

from src.chatbot.modules.chroma_handler import ChromaHandler, ChromaTools

from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from langchain.python import PythonREPL
from langchain.llms import OpenAI
from langchain.llms import OpenAIChat
from langchain.agents import load_tools, initialize_agent
from src.conf import CLIENT, AGENT_CHROMA_DATA_PATH, PARENT_SRC, DATA_DIR

from langchain.tools.file_management import (
    ReadFileTool,
    CopyFileTool,
    DeleteFileTool,
    MoveFileTool,
    WriteFileTool,
    ListDirectoryTool,
)
from langchain.tools import ShellTool
from langchain.agents.agent_toolkits import FileManagementToolkit
import os
from pathlib import Path
from src.chatbot.utils import get_folders

chroma_client = CLIENT


def python_tool():
    python_repl = PythonREPL()
    repl_tool = Tool(
        name="python_repl",
        description="A Python shell. Use this to execute python commands. Input should be a valid python script. If you want to see the output of a value, you should print it out with `print(...)`.",
        func=python_repl.run,
    )
    # agent_executor = create_python_agent(
    #     llm=OpenAIChat(temperature=0.1, max_tokens=1000),
    #     tool=repl_tool,
    #     verbose=True,
    #     max_iteration=4,
    # )
    # tool = Tool(
    #     name="Python_REPL Agent",
    #     description="A Python developer capable of implementing logic in python. Ask this agent to execute tasks in python shell. Input should be a valid task with proper instructions wrt to libraries to be used .",
    #     func=lambda x: agent_executor.run(x),
    # )
    return repl_tool


def more_tool():
    chat_llm = OpenAIChat(temperature=0.0)
    tools = load_tools(
        ["ddg-search"],
        llm=chat_llm,
    )
    return tools


def load_context_document_tool(context="test"):
    save_dir = AGENT_CHROMA_DATA_PATH
    croma_handler = ChromaTools(save_dir, chroma_client, context=context)
    if len(croma_handler.get_all_indexed_files()) > 0:
        tool1 = croma_handler.get_langchain_tool()
        tool2 = croma_handler.get_file_addition_tool()
        tool3 = croma_handler.get_file_deletion_tool()
        return [tool1, tool2, tool3]

    else:
        return None


def load_listfiles_tools(
    working_dir=DATA_DIR, folders=get_folders(DATA_DIR, exclude=["CSV"])
):
    assert os.path.isdir(working_dir) == True, f"No directiory exists :{working_dir}"
    if type(folders) == list and len(folders) > 0:
        all_tools = []
        for fold in folders:
            new_working_dir = fold.split("/")[-1]
            tool = FileManagementToolkit(
                root_dir=fold,
                selected_tools=["list_directory"],
            ).get_tools()[0]
            tool.name = f"List_{new_working_dir}_dir"
            tool.description += (
                f"Current directory you have access to is {PARENT_SRC}/{fold}"
            )
            all_tools.append(tool)

        return all_tools
    else:
        tool = FileManagementToolkit(
            root_dir=fold,
            selected_tools=["list_directory"],
        ).get_tools()[0]

        tool[
            0
        ].description += (
            f"Current directory you have access to is {PARENT_SRC}/{working_dir}"
        )
        return tool


def load_shell_tool():
    shell_tool = ShellTool()
    shell_tool.description = shell_tool.description + f"args {shell_tool.args}".replace(
        "{", "{{"
    ).replace("}", "}}")
    return [shell_tool]


def load_all_tools(context="test"):
    python_tools = python_tool()
    all_tools = more_tool()
    all_tools.append(python_tools)

    listfiles_tools = load_listfiles_tools()
    all_tools = all_tools + listfiles_tools

    shell_tool = load_shell_tool()
    all_tools = all_tools + shell_tool

    doc_tools = load_context_document_tool(context)
    all_tools = all_tools + doc_tools

    return all_tools
