import time
from typing import Any
from langchain.tools import BaseTool

from app.utils.duckdb_query import run_sql


class DuckDBTool(BaseTool):
    name = "execute"
    description = """useful for when you need to run SQL queries against a DuckDB database.
    Input to this tool is a detailed and correct SQL query, output is a result from the database.
    If the query is not correct, an error message will be returned.
    If an error is returned, rewrite the query, check the query, and try again.
    """

    duckdb_connection: Any = None

    def __init__(self, duckdb_connection, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.duckdb_connection = duckdb_connection

    def _run(self, query: str) -> str:
        query_result = run_sql(self.duckdb_connection, query)
        time.sleep(1)
        return query_result

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Not supported")
