from langchain.tools import BaseTool


class SyncTool(BaseTool):
    # these are tools are not async
    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("Cannot be executed async")


class UserInputTool(SyncTool):
    name = "user_input"
    description = "Query the user for input or an opinion, only as a last resort."

    def _run(self, query: str) -> str:
        print(query)
        return input(">>> ")