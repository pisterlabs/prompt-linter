"""Tool that run shell commands."""
import asyncio
import logging
import re
import subprocess
from typing import List, Optional, Type

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool, ToolException
from openai.error import OpenAIError
from pydantic import BaseModel, Field, validator

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)


class ShellCommandHistory:
    """
    A class to manage the history of shell commands and their outputs.

    Attributes
    ----------
    last_commands : List[Tuple[str, str]]
        A list to save the pairs of command and output.
    """

    def __init__(self):
        """Initializes ShellCommandHistory with an empty command list."""
        self.last_commands = []

    def add_command(self, command, output):
        """
        Adds a command and output pair to the command list.

        Parameters
        ----------
        command : str
            The shell command executed.
        output : str
            The output from the command.
        """
        self.last_commands.append((command, output))

    def get_last_commands(self):
        """
        Returns the last commands and output pairs.

        Returns
        -------
        list
            A list of tuples, where each tuple contains a command and its output.
        """
        return self.last_commands

    def clear(self):
        """Clears the command list."""
        self.last_commands = []


class ShellToolSchema(BaseModel):
    """
    Schema to validate shell commands.

    Attributes
    ----------
    commands: List[str]
        List of shell commands to be executed
    """

    commands: List[str] = Field(
        description="List of commands to run in a bash shell session"
    )


class ShellTool(BaseTool):
    """Tool to run shell commands."""

    name: str = "sh"
    """Name of tool."""

    description: str = (
        "Run shell commands on this machine. "
        "Each command is run in the same shell session, so commands "
        "can affect subsequent ones. Outputs (stdout and stderr) are captured."
    )
    """Description of tool."""

    args_schema: Type[BaseModel] = ShellToolSchema
    """Schema for input arguments."""

    process: Optional[subprocess.Popen] = None
    eof_marker: str = "<EOF_MARKER>"
    command_history: ShellCommandHistory = ShellCommandHistory()
    show_output: bool = False

    def _create_process(self):
        return subprocess.Popen(
            "/bin/bash",
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def toggle_output(self):
        """
        Toggles the output display of the shell command.

        If the output display is currently on, it will be turned off.
        If the output display is currently off, it will be turned on.
        """
        self.show_output = not self.show_output
        print(f"Output is now {'ON' if self.show_output else 'OFF'}.")

    def get_working_directory(self) -> str:
        """
        Returns the current working directory of the subprocess.

        This method executes the `pwd` command in the subprocess and then captures and
        returns the output, which is the current working directory.

        Returns
        -------
        str
            The current working directory of the subprocess.
        """
        command = "pwd"
        self.process.stdin.write(command + "\n")
        self.process.stdin.write("echo\n")
        self.process.stdin.write('echo "{}"\n'.format(self.eof_marker))
        self.process.stdin.flush()
        output = ""
        for line in iter(self.process.stdout.readline, ""):
            if line.strip() == self.eof_marker:
                break
            output += line
        return output.strip()

    def _run(
        self,
        commands: List[str],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        try:
            for command in commands:
                print(f"$ {command}")
                self.process.stdin.write(command + " < /dev/null\n")
            self.process.stdin.write("echo\n")
            self.process.stdin.write('echo "{}"\n'.format(self.eof_marker))
            self.process.stdin.flush()
            output = ""

            for line in iter(self.process.stdout.readline, ""):
                if line.strip() == self.eof_marker:
                    break
                output += line
                if self.show_output:
                    print(line, end="")
            output = output.strip()
            self.command_history.add_command(commands, output)
            return output[: (4096 // 4)]
        except (OpenAIError, IOError) as e:
            logging.error(str(e), exc_info=True)
            raise ToolException(str(e)) from e

    async def _arun(
        self,
        commands: List[str],
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Run commands asynchronously and return final output."""
        return await asyncio.get_event_loop().run_in_executor(None, self._run, commands)

    def close(self):
        """
        Closes the shell process.

        This method closes the standard input of the shell process and waits for the process to exit.
        It should be called when the ShellTool object is no longer needed.
        """
        self.process.stdin.close()
        self.process.terminate()
        self.process.wait(timeout=0.2)

    def __enter__(self):
        self.process = self._create_process()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
