import sqlite3

from langchain.tools import Tool
from ecommerce_agent.sql_query_schema.run_query_args_schema import RunQueryArgsSchema

conn = sqlite3.connect('db.sqlite')


# Run a SQLite query
def run_sqlite_query(query):
    cursor = conn.cursor()
    try:
        cursor.execute(query)
        return cursor.fetchall()
    except sqlite3.OperationalError as err:
        return f"The following error occurred: {str(err)}"


# Tool to run the SQL query using the run_sqlite_query function
sql_query_tool = Tool.from_function(name="sql_query_runner",
                                    description="Run a SQLite query.",
                                    func=run_sqlite_query,
                                    args_schema=RunQueryArgsSchema)
