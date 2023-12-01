from langchain.agents import Tool
import typing


class SimpleTool:
    name: str
    description: str
    func: typing.Callable[[str], str]

    def get_tool(self):
        return Tool(name=self.name, func=self.func, description=self.description)


class WarningTool(SimpleTool):
    name: str = "WarnAgent"
    description: str = "A tool that can be used to warn the agent about something."

    @staticmethod
    def func(args: str) -> str:
        return '\r' + args + '\n'
