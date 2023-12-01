import pandas as pd

from langchain.tools import BaseTool


class DropColumnTool(BaseTool):
    name = "drop_column_tool"
    description = """This is useful for deleting a column. The input is the column name."""

    def _run(self, query: str) -> str:
        """Use the tool."""

        df = pd.read_csv('/home/langchain-tools/data/sample2.csv')
        df = df.drop(query, axis=1)
        df.to_csv('/home/langchain-tools/data/sample2.csv', index=False)

        result = "Deletion of columns is complete." 
        return result

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")