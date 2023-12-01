import sqlite3
from pydantic.v1 import BaseModel
from typing import List
from langchain.tools import Tool

conn = sqlite3.connect('db.sqlite')


def list_tables():
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    rows = c.fetchall()
    return "\n".join(r[0] for r in rows if r[0] is not None)


def run_sqlite_query(query):
    c = conn.cursor()
    try:
        c.execute(query)
        return c.fetchall()
    except sqlite3.OperationalError as err:
        return f"The following error occurred: {str(err)}"


class RunQueryArgsSchema(BaseModel):
    query: str


run_query_tool = Tool.from_function(
    name="run_sqlite_query",
    description="Run a sqlite query.",
    func=run_sqlite_query,
    args_schema=RunQueryArgsSchema
)


def describe_tables(table_names):
    c = conn.cursor()
    tables = ', '.join("'{}'".format(t) for t in table_names)
    rows = c.execute(
        f"SELECT sql FROM sqlite_master WHERE type='table' and name IN ({tables});")
    return "\n".join(r[0] for r in rows if r[0] is not None)


class DescribeTablesArgsSchema(BaseModel):
    table_names: List[str]


describe_tables_tool = Tool.from_function(
    name="describe_tables",
    description="Given a list of table names, returns the schema of those tables.",
    func=describe_tables,
    args_schema=DescribeTablesArgsSchema
)
