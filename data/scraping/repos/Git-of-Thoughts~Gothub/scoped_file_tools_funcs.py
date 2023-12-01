import os
import subprocess
from typing import Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool
from langchain.tools.file_management import ReadFileTool
from langchain.tools.file_management.utils import (
    INVALID_PATH_TEMPLATE,
    BaseFileToolMixin,
    FileValidationError,
)
from pydantic import BaseModel, Field


class MyCreateToolInput(BaseModel):
    """Input for FileTool."""

    file_path: str = Field(..., description="Path of the file, include file name")


class MyFillToolInput(BaseModel):
    """Input for FileTool."""

    content: str = Field(
        ..., description="content to write to file, excluding file path"
    )


class MyLocateToolInput(BaseModel):
    """Input for LocateTool."""

    line_number: str = Field(
        ...,
        description=(
            "Line number to locate where edits take place, "
            "remember to strictly input in the format of num1:num2."
        ),
    )


class MyEditLineToolInput(BaseModel):
    """Input for EditLineTool."""

    content: str = Field(
        ..., description="content to write to the edited file, excluding file path"
    )


class MyScriptExecutionToolInput(BaseModel):
    """Input for TestScriptFileTool."""

    script_path: str = Field(..., description="Path of the script to be tested")


def file_tool_factory():
    prev_file_path = None

    class MyCreateFileTool(BaseFileToolMixin, BaseTool):
        name: str = "create_file_tool"
        args_schema: Type[
            BaseModel
        ] = MyCreateToolInput  # Accepts a single string argument
        description: str = "Create a new file"

        def _run(self, file_path: str) -> str:
            file_path = self.get_relative_path(file_path)  # root_dir + "/file_path"

            nonlocal prev_file_path
            prev_file_path = file_path

            # TODO append functionality
            append = False

            if os.path.exists(file_path):
                return f"Writing existing file: {file_path}"

            # creating a new file, path doesn't exist
            try:
                mode = "a" if append else "w"
                with open(file_path, mode) as f:
                    f.write("created successfully")
                return f"File created successfully to {file_path}."
            except Exception as e:
                return "Error: " + str(e)

        async def _arun(self) -> str:
            # TODO: Add aiofiles method
            raise NotImplementedError

    class MyFillFileTool(BaseFileToolMixin, BaseTool):
        name: str = "write_file_tool"
        args_schema: Type[BaseModel] = MyFillToolInput
        description: str = "Write to a file"

        def _run(self, content: str) -> str:
            try:
                nonlocal prev_file_path
                file_path = prev_file_path
                file_path = self.get_relative_path(file_path)  # root_dir + "/file_path"

                with open(file_path, "w") as file:
                    file.write(content)
                return f"File content written successfully to {file_path}."

            except Exception as e:
                return "Error: " + str(e)

        async def _arun(self) -> str:
            # TODO: Add aiofiles method
            raise NotImplementedError

    return MyCreateFileTool, MyFillFileTool


def edit_file_tool_factory():
    prev_file_path = None
    edit_line_num = None

    class MyReadLineTool(ReadFileTool):
        name: str = "read_line_tool"
        args_schema: Type[
            BaseModel
        ] = MyCreateToolInput  # Accepts a single string argument
        description: str = "Read a file to find line numbers to edit"

        def _run(
            self,
            file_path: str,
        ) -> str:
            try:
                read_path = self.get_relative_path(file_path)
            except FileValidationError:
                return INVALID_PATH_TEMPLATE.format(
                    arg_name="file_path", value=file_path
                )
            if not read_path.exists():
                return f"Error: no such file or directory: {file_path}"
            try:
                with read_path.open("r", encoding="utf-8") as f:
                    content = f.readlines()
                    nonlocal prev_file_path
                    prev_file_path = read_path
                return content
            except Exception as e:
                return "Error: " + str(e)

        async def _arun(self) -> str:
            # TODO: Add aiofiles method
            raise NotImplementedError

    class MyLocateLineTool(ReadFileTool):
        name: str = "locate_line_tool"
        args_schema: Type[BaseModel] = MyLocateToolInput  # Accepts a line range string
        description: str = "Locate and validate line numbers in a file"

        def _run(self, line_range: str) -> str:
            try:
                start, end = map(int, line_range.split(":"))
            except ValueError:
                return "Error: input can only be in the format of num1:num2"

            # Reading the file again is not very efficient,
            # but necessary to validate line numbers
            with prev_file_path.open("r", encoding="utf-8") as f:
                lines = f.readlines()

            if start < 0 or end > len(lines):
                return f"Error: line range {start}:{end} is out of file bounds"

            nonlocal edit_line_num
            edit_line_num = (start, end)  # Save the line numbers for the next tool
            return "Success: line numbers are valid"

        async def _arun(self) -> str:
            # TODO: Add aiofiles method
            raise NotImplementedError

    class MyEditLineTool(BaseFileToolMixin, BaseTool):
        name: str = "edit_line_tool"
        args_schema: Type[
            BaseModel
        ] = MyEditLineToolInput  # Accepts a line range string
        description: str = "edit contents in a file at specific line numbers"

        def _run(self, content: str) -> str:
            nonlocal edit_line_num
            if edit_line_num is None:
                return "Error: No line numbers set. Run MyLocateLineTool first."
            start, end = edit_line_num
            try:
                with prev_file_path.open("r", encoding="utf-8") as f:
                    lines = f.readlines()
                # Edit the lines
                lines[start : end + 1] = content

                with prev_file_path.open("w", encoding="utf-8") as f:
                    f.writelines(lines)
                return f"Edited file {prev_file_path} at lines {start}:{end}"
            except Exception as e:
                return "Error: " + str(e)

        async def _arun(self) -> str:
            # TODO: Add aiofiles method
            raise NotImplementedError

    return MyReadLineTool, MyLocateLineTool, MyEditLineTool


class MyScriptExecutionTool(BaseFileToolMixin, BaseTool):
    name: str = "test_execution_tool"
    args_schema: Type[BaseModel] = MyScriptExecutionToolInput
    description: str = "Execute the provided script as a test."

    def _run(self, script_path: str) -> str:
        try:
            script_path = self.get_relative_path(script_path)
            # Determine the type of script and set the command accordingly
            if script_path.endswith(".py"):
                cmd = ["python", script_path]
            else:
                return f"Error: Unsupported script type for file {script_path}"

            # Run the script
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                return f"Error: Script execution failed with message {result.stderr}"
            return f"Script {script_path} executed successfully"
        except Exception as e:
            return f"Error: {str(e)}"

    async def _arun(self) -> str:
        # TODO: Add asyncio subprocess execution method
        raise NotImplementedError
