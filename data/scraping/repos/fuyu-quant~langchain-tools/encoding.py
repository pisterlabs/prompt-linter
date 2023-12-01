# One-hot　エンコーディング
# Label エンコーディング
# Target エンコーディング

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder

from langchain.tools import BaseTool


class OnehotEncodingTool(BaseTool):
    name = "onehot_encoding_tool"
    description = """It is useful for onehot encoding when column names are given. The input should be a column name."""

    def _run(self, query: str) -> str:
        """Use the tool."""

        df = pd.read_csv('/home/langchain-tools/data/sample2.csv')
        df = pd.get_dummies(df, columns=[query])
        df.to_csv('/home/langchain-tools/data/sample2.csv', index=False)

        result = "one-hot encoding has been completed. " 
        return result

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")




class LabelEncodingTool(BaseTool):
    name = "label_encoding_tool"
    description = """It is useful for label encoding when column names are given. The input should be a column name."""

    def _run(self, query: str) -> str:
        """Use the tool."""

        df = pd.read_csv('/home/langchain-tools/data/sample2.csv')
        le = LabelEncoder()
        df[query] = le.fit_transform(df[query])
        df.to_csv('/home/langchain-tools/data/sample2.csv', index=False)

        result = "label encoding has been completed. " 
        return result

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")




class TargetEncodingTool(BaseTool):
    name = "target_encoding_tool"
    description = """It is useful for target encoding when column names are given. The input receives the column names of the target column and the column name of the objective variable, separated by commas, such as 'A,B'."""

    def _run(self, query: str) -> str:
        """Use the tool."""
        a, b = query.split(",")

        df = pd.read_csv('/home/langchain-tools/data/sample2.csv')
        te = TargetEncoder()
        df[a] = df[a].astype(str)
        df[a] = te.fit_transform(df[a], df[b])
        df.to_csv('/home/langchain-tools/data/sample2.csv', index=False)

        result = "target encoding has been completed. " 
        return result

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")