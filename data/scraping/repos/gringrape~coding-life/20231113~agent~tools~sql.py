import sqlite3

from langchain.tools import Tool, StructuredTool

from pydantic.v1 import BaseModel

connection = sqlite3.connect("data/db.sqlite")


def write_report(filename, html):
    with open(filename, "w") as f:
        f.write(html)


def list_tables():
    cursor = connection.cursor()
    rows = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = (row[0] for row in rows if row[0] is not None)
    return "\n".join(tables)


def describe_table(table_name: str):
    cursor = connection.cursor()
    rows = cursor.execute(f"PRAGMA table_info({table_name});")
    columns = (row[1] for row in rows if row[1] is not None)
    return ", ".join(columns)


def run_sqlite_query(query):
    cursor = connection.cursor()

    try:
        cursor.execute(query)
        return cursor.fetchall()
    except sqlite3.OperationalError as err:
        return f"the following error occured{str(err)}"


class DescribeToolSchema(BaseModel):
    table_name: str


describe_table_tool = Tool.from_function(
    name="describe_table",
    description="Retrieves column names of given table",
    func=describe_table,
    args_schema=DescribeToolSchema,
)


class RunQuerySchema(BaseModel):
    query: str


run_query_tool = Tool.from_function(
    name="run_sqlite_query",
    description="Run a sqlite query",
    func=run_sqlite_query,
    args_schema=RunQuerySchema,
)


class WriteReportSchema(BaseModel):
    filename: str
    html: str


write_report_tool = StructuredTool.from_function(
    name="write_report",
    description="Write html report. Use this whenever user request to write report ",
    func=write_report,
    args_schema=WriteReportSchema,
)
