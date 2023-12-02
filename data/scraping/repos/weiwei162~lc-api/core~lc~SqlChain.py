from langchain.prompts.prompt import PromptTemplate
from langchain.output_parsers.list import CommaSeparatedListOutputParser
from langchain.schema.runnable import RunnableMap
from operator import itemgetter

from langchain.schema.language_model import BaseLanguageModel
from langchain import SQLDatabase

from sqlalchemy import create_engine, text
import pandas as pd
from core.lc.nc import get_db_uri


_DEFAULT_TEMPLATE = """Given an input question, first create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer. Unless the user specifies in his question a specific number of examples he wishes to obtain, always limit your query to at most {top_k} results. You can order the results by a relevant column to return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

{table_info}

Question: {input}"""
PROMPT = PromptTemplate.from_template(_DEFAULT_TEMPLATE)

_DECIDER_TEMPLATE = """Given the below input question and list of potential tables, output a comma separated list of the table names that may be necessary to answer this question.

Question: {query}

Table Names: {table_names}

Relevant Table Names:"""
DECIDER_PROMPT = PromptTemplate.from_template(_DECIDER_TEMPLATE)

final_prompt_template = """Based on the table schema below, question, sql query, and sql response, write a natural language response in Chinese:
Question: {question}
SQL Query: {query}
SQL Response: {response}"""
final_prompt = PromptTemplate.from_template(final_prompt_template)


def sql_query(uri, sql):
    engine = create_engine(uri)
    with engine.connect() as conn:
        df = pd.read_sql_query(text(sql), conn)
        return df.to_dict('records')


class SqlChain():
    def from_llm(
        llm: BaseLanguageModel,
        ds_id: str,
        k: int = 5,
    ) -> any:
        uri = get_db_uri(ds_id)

        db = SQLDatabase.from_uri(
            uri,
            # {'echo': True},
            # sample_rows_in_table_info=0
        )

        _table_names = db.get_usable_table_names()
        table_names = ", ".join(_table_names)

        table_names_from_chain = {"query": itemgetter(
            "question"), "table_names": lambda _: table_names} | DECIDER_PROMPT | llm | CommaSeparatedListOutputParser()

        inputs = {
            "input": lambda x: x["question"] + "\nSQLQuery: ",
            "top_k": lambda _: k,
            "table_info": lambda x: db.get_table_info(
                table_names=x.get("table_names_to_use")
            ),
            "dialect": lambda _: db.dialect
        }

        sql_response = inputs | PROMPT | llm.bind(stop=["\nSQLResult:"])

        final_result = {"question": itemgetter("question"), "query": itemgetter(
            "query"), "response": lambda x: db.run(x["query"])} | final_prompt | llm

        return (
            RunnableMap({
                "question": itemgetter("question"),
                "table_names_to_use": table_names_from_chain,
            })
            | RunnableMap({
                "question": itemgetter("question"),
                "query": sql_response,
            })
            | RunnableMap({
                "sql": itemgetter("query"),
                "data": lambda x: sql_query(uri, x["query"]),
                "result": final_result,
            })
        )
