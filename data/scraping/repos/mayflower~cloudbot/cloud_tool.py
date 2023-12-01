import asyncio
import platform
from typing import List, Optional, Type, Union

from pydantic import BaseModel, Field, root_validator

from langchain.callbacks.manager import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain.tools.base import BaseTool
from langchain.utilities.bash import BashProcess


class CloudInput(BaseModel):
    """Commands for Cloud tool."""

    commands: str = Field(
        ...,
        description="List of shell commands to run. Deserialized using json.loads",
    )
    """List of shell commands to run."""

    @root_validator
    def _validate_commands(cls, values: dict) -> dict:
        """Validate commands."""
        command = values.get("commands")
        cmd_name = command.split()[0]
        if cmd_name in ["kubectl", "aws", "helm"]:
            values["commands"] = command
        else:
            raise ValueError(f"Sorry, command {cmd_name} is not permitted.")
        return values


def _get_default_bash_processs() -> BashProcess:
    """Get file path from string."""
    return BashProcess(return_err_output=True)


def _get_platform() -> str:
    """Get platform."""
    system = platform.system()
    if system == "Darwin":
        return "MacOS"
    return system


class CloudTool(BaseTool):
    """Tool to run shell commands."""

    process: BashProcess = Field(default_factory=_get_default_bash_processs)
    """Bash process to run commands."""

    name: str = "terminal"
    """Name of tool."""

    description: str = f"Run shell commands on this {_get_platform()} machine."
    """Description of tool."""

    args_schema: Type[BaseModel] = CloudInput
    """Schema for input arguments."""

    def is_single_input(self) -> bool:
        return True

    def _run(
        self,
        commands: Union[str, List[str]],
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Run commands and return final output."""
        return self.process.run(commands)

    async def _arun(
        self,
        commands: Union[str, List[str]],
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Run commands asynchronously and return final output."""
        return await asyncio.get_event_loop().run_in_executor(
            None, self.process.run, commands
        )
