from langchain.tools.file_management import (
    ReadFileTool,
    CopyFileTool,
    DeleteFileTool,
    MoveFileTool,
    WriteFileTool,
    ListDirectoryTool,
)
from langchain.agents.agent_toolkits import FileManagementToolkit
from tempfile import TemporaryDirectory

# We'll make a temporary directory to avoid clutter
working_directory = TemporaryDirectory(dir="../../data/gptsql/")
toolkit = FileManagementToolkit(root_dir=str(working_directory.name)) # If you don't provide a root_dir, operations will default to the current working directory
toolkit.get_tools()

tools = FileManagementToolkit(root_dir=str(working_directory.name), selected_tools=["read_file", "write_file", "list_directory"]).get_tools()
print(tools)


read_tool, write_tool, list_tool = tools
result_write_operation = write_tool.run({"file_path": "example_v000.txt", "text": "Hello World!"})

# List files in the working directory
print("A list of files in the working dir:\n"+list_tool.run({}))


