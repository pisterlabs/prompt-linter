import pandas as pd
from const import INPUT_DIR
from langchain.tools import BaseTool
from utils import text_processing


class DropColumnTool(BaseTool):
    name = "drop_column_tool"
    description = """This is useful for deleting a column. The input should be the column name."""

    def _run(self, query: str) -> str:
        """Use the tool."""
        query = text_processing(query)

        df = pd.read_csv(f"{INPUT_DIR}/stored_df.csv")
        df = df.drop(query, axis=1)
        df.to_csv(f"{INPUT_DIR}/stored_df.csv", index=False)

        result = f"{self.name} Deletion of columns {query} is complete."

        return result

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")
