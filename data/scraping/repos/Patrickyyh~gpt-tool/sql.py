import sqlite3
from langchain.tools import Tool
from pydantic.v1 import BaseModel
from typing import List



conn = sqlite3.connect('db.sqlite')


def list_tables():
    c = conn.cursor()
    c.execute("SELECT name FROM sqlite_master WHERE type='table';")
    rows = c.fetchall()
    return ("\n".join(row[0] for row in rows if row[0] is not None))



def run_sqlite_query(query):
    c = conn.cursor()
    try:
        c.execute(query)
        return c.fetchall()
    except sqlite3.OperationalError as e:
        return f"The following error occured: {str(e)}"


class RunQueryToolInput(BaseModel):
    query: str


run_query_tool = Tool.from_function(
    name = "run_sqlite_query",
    description= "Run a sqlite query.",
    func = run_sqlite_query,
    args_schema= RunQueryToolInput
)

def describe_tables(table_names):
    c = conn.cursor()
    tables = ', '.join( "'" + table + "'" for table in table_names)
    rows = c.execute(f"SELECT sql FROM sqlite_master WHERE type='table' AND name IN ({tables});")
    return "\n".join(row[0] for row in rows if row[0] is not None)


class DescribeTablesSchema(BaseModel):
    tables_names: List[str]


describe_tables_tool = Tool.from_function(
    name = "describe_tables",
    description= "Given a list of table names, return the schema of the tables.",
    func = describe_tables,
    args_schema=DescribeTablesSchema
)



