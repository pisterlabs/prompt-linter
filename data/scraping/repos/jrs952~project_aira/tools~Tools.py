# define a function to set up Github Tools
from langchain.agents import Tool
from .GithubManager import GithubManager
def setup_tools(ghManager: GithubManager):
    tools = [
        Tool.from_function(
            name=ghManager.creator_tool_name,
            description=ghManager.creator_tool_description,
            func=ghManager.create_repo
        ),
        Tool.from_function(
            name=ghManager.addFile_tool_name,
            description=ghManager.addFile_tool_description,
            func=ghManager.add_file
        ),
        Tool.from_function(
            name=ghManager.updateFile_tool_name,
            description=ghManager.updateFile_tool_description,
            func=ghManager.update_file
        ),
            Tool.from_function(
            name=ghManager.lookup_repo_tool_name,
            description=ghManager.lookup_repo_tool_description,
            func=ghManager.repo_lookup
        )
    ]
    return tools
