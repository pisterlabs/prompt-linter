# 欠損値を0で埋める
# 欠損値を平均で埋める
# 欠損値を中央値で埋める
import pandas as pd

from langchain.tools import BaseTool


class File0Tool(BaseTool):
    name = "file_0_tool"
    description = """It is useful to fill in missing values with zeros when a column name is received. The input should be a column name."""

    def _run(self, query: str) -> str:
        """Use the tool."""

        df = pd.read_csv('/home/langchain-tools/data/sample2.csv')
        df[query] = df[query].fillna(0)
        df.to_csv('/home/langchain-tools/data/sample2.csv', index=False)

        result = "The missing values could be filled with 0." 
        return result

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")



class FileMeansTool(BaseTool):
    name = "file_means_tool"
    description = """It is useful to fill in missing values with means when a column name is received. The input should be a column name."""

    def _run(self, query: str) -> str:
        """Use the tool."""

        df = pd.read_csv('/home/langchain-tools/data/sample2.csv')
        df[query] = df[query].fillna(df[query].mean())
        df.to_csv('/home/langchain-tools/data/sample2.csv', index=False)

        result = "The missing values could be filled with means." 
        return result

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")



class FileMedianTool(BaseTool):
    name = "file_median_tool"
    description = """It is useful to fill in missing values with median when a column name is received. The input should be a column name."""

    def _run(self, query: str) -> str:
        """Use the tool."""

        df = pd.read_csv('/home/langchain-tools/data/sample2.csv')
        df[query] = df[query].fillna(df[query].median())
        df.to_csv('/home/langchain-tools/data/sample2.csv', index=False)

        result = "The missing values could be filled with median." 
        return result

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")