from typing import Optional

import pandas as pd
from langchain.callbacks.manager import CallbackManagerForToolRun, AsyncCallbackManagerForToolRun
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.tools import tool, BaseTool
from langchain.vectorstores.chroma import Chroma
from sqlalchemy import create_engine
from dotenv import load_dotenv

# 加载.env文件中的环境变量
load_dotenv()


class CustomTrinoTableSchema(BaseTool):
    name = "get_trino_table_schema"
    description = """useful for when you need to get the schema of the table.Before that you should calling get_trino_list_table to get used table
The input to this tool should be a full stop separated string.
For example, database1.table1 would be the input if you wanted to get the schema of the table table1 in the database database1."""

    def _run(
            self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        database, table_name = query.split(".")
        engine = create_engine('trino+pyhive://')
        df = pd.read_sql_query(f'DESC {database}.{table_name}', engine)
        result_schema = self.get_df_str(df)
        df_sample = pd.read_sql_query(f'select * from {database}.{table_name} limit 1', engine)
        result_sample = self.get_df_str(df_sample)
        result_final = result_schema + f'\n\nsample rows from {database}.{table_name} table:\n' + result_sample
        return result_final

    async def _arun(
            self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        database, table_name = query.split(".")
        engine = create_engine('trino+pyhive://')
        df = pd.read_sql_query(f'DESC {database}.{table_name}', engine)
        result_schema = self.get_df_str(df)
        df_sample = pd.read_sql_query(f'select * from {database}.{table_name} limit 1', engine)
        result_sample = self.get_df_str(df_sample)
        result_final = result_schema + f'\n\nsample rows from {database}.{table_name} table:\n' + result_sample
        return result_final

    def get_df_str(self, df):
        # 获取列名并打印
        column_names = df.columns.tolist()
        column_names_str = ' '.join(column_names)

        # 打印每一行的内容并拼接
        rows = []
        for _, row in df.iterrows():
            row_values = [str(row[column]) for column in column_names]
            rows.append(' '.join(row_values))

        result_rows = '\n'.join(rows)
        result_schema = column_names_str + '\n' + result_rows
        return result_schema


class CustomTrinoSqlQuery(BaseTool):
    name = "get_trino_sql_query"
    description = """Input to this tool is a detailed and correct SQL query, output is a result from the database. 
If the query is not correct, an error message will be returned. If an error is returned, rewrite the query, check the query, and try again.
If you encounter an issue with Unknown column 'xxxx' in 'field list', using get_trino_table_schema to query the correct table fields.
Note :
get data formatting from sample rows from table."""

    def _run(
            self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        try:
            engine = create_engine('trino+pyhive://')
            df_sample = pd.read_sql_query(f'select * from ({query})t1 limit 5', engine)
            result_sample = self.get_df_str(df_sample)
            result_final = result_sample
            return result_final
        except Exception as e:
            return str(e)

    async def _arun(
            self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        try:
            engine = create_engine('trino+pyhive://')
            df_sample = pd.read_sql_query(f'select * from ({query})t1 limit 3', engine)
            result_sample = self.get_df_str(df_sample)
            result_final = result_sample
            return result_final
        except Exception as e:
            return str(e)

    def get_df_str(self, df):
        # 获取列名并打印
        column_names = df.columns.tolist()
        column_names_str = ' '.join(column_names)

        # 打印每一行的内容并拼接
        rows = []
        for _, row in df.iterrows():
            row_values = [str(row[column]) for column in column_names]
            rows.append(' '.join(row_values))

        result_rows = '\n'.join(rows)
        result_schema = column_names_str + '\n' + result_rows
        return result_schema


class CustomTrinoSqlCheck(BaseTool):
    name = "get_trino_sql_check"
    description = """Use this tool to double check if your query is correct before executing it. 
Always use this tool before executing a query with get_trino_sql_query!"""

    def _run(
            self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        try:
            engine = create_engine('trino+pyhive://')
            pd.read_sql_query(f'explain {query}', engine)
            return "pass"
        except Exception as e:
            return "failed"

    async def _arun(
            self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        try:
            engine = create_engine('trino+pyhive://')
            pd.read_sql_query(f'explain {query}', engine)
            return "pass"
        except Exception as e:
            return "failed"


class CustomTrinoListTable(BaseTool):
    name = "get_trino_list_table"
    description = """Input to this tool should be table comment in the user question,
output is a comma separated list of table name about the table comment."""

    def _run(
            self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        # Load the document, split it into chunks, embed each chunk and load it into the vector store.
        path = r'C:\Users\lilong\PycharmProjects\chainlit\source\hive_table_list.txt'
        loader = TextLoader(path, encoding='utf-8')

        # 用文件缓存已经存在的向量，后续可以翻到ES上加速，大量的存贮
        underlying_embeddings = OpenAIEmbeddings()
        fs = LocalFileStore("./test_cache/hive_table_list/")

        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings, fs, namespace=underlying_embeddings.model
        )
        # 用分号切割文件
        text_splitter = CharacterTextSplitter(
            separator=";",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        # 可以使用向量检索或其他搜索算法来实现搜索逻辑
        index_creator = VectorstoreIndexCreator(
            vectorstore_cls=Chroma,
            embedding=cached_embedder,
            text_splitter=text_splitter
        )
        index = index_creator.from_loaders([loader])
        doc_retriever = index.vectorstore.as_retriever()
        retrieved_docs = doc_retriever.invoke("return comma separated list of table name about " + query)
        return retrieved_docs[0].page_content

    async def _arun(
            self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        # Load the document, split it into chunks, embed each chunk and load it into the vector store.
        path = r'C:\Users\lilong\PycharmProjects\chainlit\source\hive_table_list.txt'
        loader = TextLoader(path, encoding='utf-8')

        # 用文件缓存已经存在的向量，后续可以翻到ES上加速，大量的存贮
        underlying_embeddings = OpenAIEmbeddings()
        fs = LocalFileStore("./test_cache/hive_table_list/")

        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings, fs, namespace=underlying_embeddings.model
        )
        # 用分号切割文件
        text_splitter = CharacterTextSplitter(
            separator=";",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        # 可以使用向量检索或其他搜索算法来实现搜索逻辑
        index_creator = VectorstoreIndexCreator(
            vectorstore_cls=Chroma,
            embedding=cached_embedder,
            text_splitter=text_splitter
        )
        index = index_creator.from_loaders([loader])
        doc_retriever = index.vectorstore.as_retriever()
        retrieved_docs = doc_retriever.invoke("return comma separated list of table name about " + query)
        return retrieved_docs[0].page_content


class CustomTrinoTableJoin(BaseTool):
    name = "get_trino_table_join"
    description = """Usefull when you need two tables join columns.
Input to this tool should be comma separated list of table name,just like db.t1,db.t2."""

    def _run(
            self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool."""
        path = r'C:\Users\lilong\PycharmProjects\chainlit\source\hive_table_join.txt'
        loader = TextLoader(path, encoding='utf-8')

        # 用文件缓存已经存在的向量，后续可以翻到ES上加速，大量的存贮
        underlying_embeddings = OpenAIEmbeddings()
        fs = LocalFileStore("./test_cache/hive_table_join/")

        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings, fs, namespace=underlying_embeddings.model
        )
        # 用分号切割文件
        text_splitter = CharacterTextSplitter(
            separator=";",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        # 可以使用向量检索或其他搜索算法来实现搜索逻辑
        index_creator = VectorstoreIndexCreator(
            vectorstore_cls=Chroma,
            embedding=cached_embedder,
            text_splitter=text_splitter
        )
        index = index_creator.from_loaders([loader])
        doc_retriever = index.vectorstore.as_retriever()
        retrieved_docs = doc_retriever.invoke("return join columns of table name about " + query)
        return retrieved_docs[0].page_content

    async def _arun(
            self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""
        path = r'C:\Users\lilong\PycharmProjects\chainlit\source\hive_table_join.txt'
        loader = TextLoader(path, encoding='utf-8')

        # 用文件缓存已经存在的向量，后续可以翻到ES上加速，大量的存贮
        underlying_embeddings = OpenAIEmbeddings()
        fs = LocalFileStore("./test_cache/hive_table_join/")

        cached_embedder = CacheBackedEmbeddings.from_bytes_store(
            underlying_embeddings, fs, namespace=underlying_embeddings.model
        )
        # 用分号切割文件
        text_splitter = CharacterTextSplitter(
            separator=";",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            is_separator_regex=False,
        )
        # 可以使用向量检索或其他搜索算法来实现搜索逻辑
        index_creator = VectorstoreIndexCreator(
            vectorstore_cls=Chroma,
            embedding=cached_embedder,
            text_splitter=text_splitter
        )
        index = index_creator.from_loaders([loader])
        doc_retriever = index.vectorstore.as_retriever()
        retrieved_docs = doc_retriever.invoke("return join columns of table name about " + query)
        return retrieved_docs[0].page_content


class CustomGenerateDateString(BaseTool):
    name = "get_date_string"
    description = """Usefull when you need get date string.
Input to this tool should be comma separated list of ,just like db.t1,db.t2."""

    def _run(
            self, query: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool"""

    async def _arun(
            self, query: str, run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> str:
        """Use the tool asynchronously."""

# c = CustomTrinoListTable()
# print(c.invoke("order_details"))
# c = CustomTrinoSqlQuery()
# print(c.invoke("select count(*) from gjdw.dim_org where dt='20231017'"))
# c = CustomTrinoSqlCheck()
# print(c.invoke("select count(*) from gjdw.dim_org where dt='20231017'"))
c = CustomTrinoTableJoin()
print(c.invoke("gjdw.dw_sale_tr_goods_dt,gjdw.dim_org"))
