from langchain.tools import ShellTool
from .parsers import parse_input
import platform

shell_tool = ShellTool()

def _get_platform() -> str:
    """Get platform."""
    system = platform.system()
    if system == "Darwin":
        return "MacOS"
    return system

def shell_tool_runner(txt) -> str:
    data = parse_input(txt)
    return shell_tool.run(data)
