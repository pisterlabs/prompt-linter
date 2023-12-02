from typing import Type
from pydantic import BaseModel, Field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple, Type, Union
from langchain.tools import BaseTool
import psycopg2
from sqlalchemy import create_engine, MetaData, Table, text
import database
import os

postgres_user=os.getenv('POSTGRES_USER')
postgres_password=os.getenv('POSTGRES_PASSWORD')
postgres_host=os.getenv('POSTGRES_HOST')
postgres_port=os.getenv('POSTGRES_PORT')
postgres_database=os.getenv('POSTGRES_DATABASE')

def execute_sql(sql: str):

    # 创建一个连接
    conn = psycopg2.connect(
        dbname=postgres_database, 
        user=postgres_user,
        password=postgres_password, 
        host=postgres_host, 
        port=postgres_port
    )

    # 创建一个游标对象
    cur = conn.cursor()
    try:
        # 执行一个SQL查询
        cur.execute(sql)
        # 获取查询结果
        rows = cur.fetchall()
        print(rows)
        # 打印查询结果
        result=f"SQL '{sql}' execute result:\n"
        for row in rows:
            result += str(row)
        return result
    except Exception as e:
        print(e)
        return f"sql '{sql} command execute with error:" +str(e)
    finally:
        # 关闭游标和连接
        cur.close()
        conn.close()

db_connection_string = ('postgresql://%s:%s@%s:%s/%s' %
                             (postgres_user, postgres_password,
                              postgres_host, postgres_port,
                              postgres_database))

def init_metadata():

    # 创建一个引擎
    engine = create_engine(db_connection_string)
    schemas= engine.connect().execute(text("SELECT schema_name FROM information_schema.schemata"))
    print(schemas)
    # 创建一个元数据对象
    metadata = MetaData()

    # 使用元数据对象反射特定schema中的表
    metadata.reflect(bind=engine, schema='public')

    # 打印所有反射的表名
    for table in metadata.tables.values():
        print(table.name)

    # 对于每个表，打印其列名和类型
    for table in metadata.tables.values():
        for column in table.c:
            print(f"Table {table.name}, Column {column.name}, Type {column.type}")






def init_all_metadata():
    schema_list=execute_sql("SELECT schema_name FROM information_schema.schemata")
    init_metadata_sql = (
        f"SELECT table_name, column_name, data_type, is_nullable, character_maximum_length"
        f"FROM information_schema.columns "
        f"WHERE table_schema = 'your_schema';"
    )

if __name__=="__main__":
    print(execute_sql("SELECT schema_name FROM information_schema.schemata"))
    init_metadata()
    
def fake_func():
    pass

class ExecutePostgressSQLInput(BaseModel):
    """Inputs for execute_sql_command"""
    sql: str = Field(description="sql_command")

class ExecutePostgressSQLTool(BaseTool):
    name = "execute_sql_command"
    description = """
        Useful when you want to need to execute a psql command to fetch data
        You should enter a psql command can be directly executed. Remember Don't use ```SELECT *``` or any other similar commond that might cause too many outputs.
        """
    args_schema: Type[BaseModel] = ExecutePostgressSQLInput
    func: Callable[..., str]=fake_func

    def _run(self, sql: str):
        response = execute_sql(sql)
        return response

    async def _arun(self, sql: str):
        response = execute_sql(sql)
        return response
        # raise NotImplementedError("get_current_stock_price does not support async")


class GetPSQLSchemaMetadataInput(BaseModel):
    """Inputs for get_psql_schema_metadata"""
    schema_name: str = Field(description="schema name of the database, nullable=false")


class GetPSQLSchemaMetadataTool(BaseTool):
    name = "get_psql_schema_metadata"
    description = """
        Useful when you need to know the metadata information of a schema
        """
    args_schema: Type[BaseModel] = GetPSQLSchemaMetadataInput
    func: Callable[..., str]=fake_func

    def _run(self, schema_name: str):
        try:
            return str(database.metadata_dict[schema_name])
        except Exception as e:
            return f"Key error with {schema_name}"

    async def _arun(self, schema_name: str):
        return str(database.metadata_dict[schema_name])
        # raise NotImplementedError("get_current_stock_price does not support async")


class GetPSQLSchemaListInput(BaseModel):
    """Inputs for get_psql_schema_list"""
    database_name: str = Field(description="database name")

class GetPSQLSchemaListTool(BaseTool):
    name = "get_psql_schema_list"
    description = """
        Useful when you need to know the all schema names of a database
        """
    args_schema: Type[BaseModel] = GetPSQLSchemaListInput
    func: Callable[..., str]=fake_func

    def _run(self, database_name: str):
        return str(database.schema_prompt)

    async def _arun(self, database_name: str):
        return str(database.schema_prompt)
        # raise NotImplementedError("get_current_stock_price does not support async")