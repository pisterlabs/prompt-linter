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
working_directory = TemporaryDirectory()


# [CopyFileTool(name='copy_file', description='Create a copy of a file in a specified location', args_schema=<class 'langchain.tools.file_management.copy.FileCopyInput'>, return_direct=False, verbose=False, callback_manager=<langchain.callbacks.shared.SharedCallbackManager object at 0x1156f4350>, root_dir='/var/folders/gf/6rnp_mbx5914kx7qmmh7xzmw0000gn/T/tmpxb8c3aug'),

# DeleteFileTool(name='file_delete', description='Delete a file', args_schema=<class 'langchain.tools.file_management.delete.FileDeleteInput'>, return_direct=False, verbose=False, callback_manager=<langchain.callbacks.shared.SharedCallbackManager object at 0x1156f4350>, root_dir='/var/folders/gf/6rnp_mbx5914kx7qmmh7xzmw0000gn/T/tmpxb8c3aug'),
# FileSearchTool(name='file_search', description='Recursively search for files in a subdirectory that match the regex pattern', args_schema=<class 'langchain.tools.file_management.file_search.FileSearchInput'>, return_direct=False, verbose=False, callback_manager=<langchain.callbacks.shared.SharedCallbackManager object at 0x1156f4350>, root_dir='/var/folders/gf/6rnp_mbx5914kx7qmmh7xzmw0000gn/T/tmpxb8c3aug'),
# MoveFileTool(name='move_file', description='Move or rename a file from one location to another', args_schema=<class 'langchain.tools.file_management.move.FileMoveInput'>, return_direct=False, verbose=False, callback_manager=<langchain.callbacks.shared.SharedCallbackManager object at 0x1156f4350>, root_dir='/var/folders/gf/6rnp_mbx5914kx7qmmh7xzmw0000gn/T/tmpxb8c3aug'),
# ReadFileTool(name='read_file', description='Read file from disk', args_schema=<class 'langchain.tools.file_management.read.ReadFileInput'>, return_direct=False, verbose=False, callback_manager=<langchain.callbacks.shared.SharedCallbackManager object at 0x1156f4350>, root_dir='/var/folders/gf/6rnp_mbx5914kx7qmmh7xzmw0000gn/T/tmpxb8c3aug'),
# WriteFileTool(name='write_file', description='Write file to disk', args_schema=<class 'langchain.tools.file_management.write.WriteFileInput'>, return_direct=False, verbose=False, callback_manager=<langchain.callbacks.shared.SharedCallbackManager object at 0x1156f4350>, root_dir='/var/folders/gf/6rnp_mbx5914kx7qmmh7xzmw0000gn/T/tmpxb8c3aug'),
# ListDirectoryTool(name='list_directory', description='List files and directories in a specified folder', args_schema=<class 'langchain.tools.file_management.list_dir.DirectoryListingInput'>, return_direct=False, verbose=False, callback_manager=<langchain.callbacks.shared.SharedCallbackManager object at 0x1156f4350>, root_dir='/var/folders/gf/6rnp_mbx5914kx7qmmh7xzmw0000gn/T/tmpxb8c3aug')]