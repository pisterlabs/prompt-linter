# One-hot　エンコーディング
# Label エンコーディング
# Target エンコーディング

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
from const import INPUT_DIR
from utils import text_processing
from langchain.tools import BaseTool


class OnehotEncodingTool(BaseTool):
    name = "onehot_encoding_tool"
    description = """It is useful for onehot encoding when column names are given. The input should be a column name."""

    def _run(self, query: str) -> str:
        """Use the tool."""
        query = text_processing(query)

        df = pd.read_csv(f"{INPUT_DIR}/stored_df.csv")
        df = pd.get_dummies(df, columns=[query])
        df.to_csv(f"{INPUT_DIR}/stored_df.csv", index=False)

        result = f"{self.name} one-hot encoding {query} has been completed. "
        return result

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")


class LabelEncodingTool(BaseTool):
    name = "label_encoding_tool"
    description = """It is useful for label encoding when column names are given. The input should be a column name."""

    def _run(self, query: str) -> str:
        """Use the tool."""
        query = text_processing(query)

        df = pd.read_csv(f"{INPUT_DIR}/stored_df.csv")
        le = LabelEncoder()
        df[query] = le.fit_transform(df[query])
        df.to_csv(f"{INPUT_DIR}/stored_df.csv", index=False)

        result = f"{self.name} label encoding {query} has been completed. "
        return result

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")


class TargetEncodingTool(BaseTool):
    name = "target_encoding_tool"
    description = """It is useful for target encoding when column names are given. The input receives the column names of the target column and the column name of the objective variable, separated by commas, such as 'A,B'."""

    def _run(self, query: str) -> str:
        """Use the tool."""
        query = text_processing(query)

        a, b = query.split(",")
        a = a.lstrip()
        b = b.lstrip()

        df = pd.read_csv(f"{INPUT_DIR}/stored_df.csv")
        te = TargetEncoder()
        df[a] = df[a].astype(str)
        df[a] = te.fit_transform(df[a], df[b])
        df.to_csv(f"{INPUT_DIR}/stored_df.csv", index=False)

        result = f"{self.name} target encoding {query} has been completed. "
        return result

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError("BingSearchRun does not support async")
