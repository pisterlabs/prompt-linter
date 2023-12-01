import json
import os
import traceback
from json import JSONDecodeError
from pathlib import Path
from typing import Optional, Union

import openai
from haystack.preview import Pipeline, component
from loguru import logger

from components import SQLExecutorComponent
from config import DB_FILE
from sql_common import EMPTY_QUERY, SQLExecutor

openai.api_key = os.environ["OPENAI_API_KEY"]


@component
class SQLAgent:
    def warm_up(self):
        self.last = False
        self.conv_history = [
            {
                "role": "system",
                "content": f"""You are an AI assistant that can query a database to answer the questions of the user.
You can perform two types of SQLite queries:
1. You can explore the database using SQLite queries to understand the schema and the general format of the database. ("show_user": false)
2. You can perform a query whose results will be shown to the user and that answers their questions. ("show_user": false)
You are supposed to execute a query of the second type only once at the end.
You cannot change a query after you have shown it to the user. Therefore, before you show a query to a user ("show_user": true), you check it on your own ("show_user": false).
This way you make sure that you have considered everything the user asked for.""",
            }
        ]  # noqa

    @component.output_types(query=str, result=list, result_query=str)
    def run(
        self, question: Optional[str] = None, result: Optional[Union[list, str]] = None
    ):
        if self.last and isinstance(result, list):
            return {
                "query": None,
                "conv_history": None,
                "last": None,
                "result": result,
                "result_query": self.query,
            }
        if question:
            self.conv_history.append({"role": "user", "content": question})
            self.query = "SELECT name FROM sqlite_schema WHERE type = 'table' AND name NOT LIKE 'sqlite_%';"
            self.conv_history.append(
                {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": "execute_query",
                        "arguments": json.dumps(
                            {"query": self.query, "show_user": False}
                        ),
                    },
                }
            )
            return {"query": self.query, "result": None, "result_query": None}
        if isinstance(result, list):
            result = repr(result)
        if result:
            self.conv_history.append(
                {"role": "function", "name": "execute_query", "content": result}
            )
        functions = [
            {
                "name": "execute_query",
                "description": "Executes an SQLite query on DB. If show_user is true the result will be shown to the user to answer their question. Otherwise, it will be returned to the assistant.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "show_user": {"type": "boolean"},
                    },
                },
            }
        ]
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=self.conv_history,
            functions=functions,
            function_call={"name": "execute_query"},
            temperature=0,
        )["choices"][0]["message"]
        logger.info(response)
        try:
            args = json.loads(response["function_call"]["arguments"])
            self.last = args.get("show_user", False)
            self.query = args["query"]
            self.conv_history.append(response)
            return {"query": self.query, "result": None, "result_query": None}
        except JSONDecodeError as e:
            # In case the response is invalid, we don't try again but
            # instead just stop the run.
            logger.warning("".join(traceback.format_tb(tb=e.__traceback__)))
            self.last = True
            return {"query": None, "result": [], "result_query": EMPTY_QUERY}


@component
class OutputSink:
    @component.output_types(output=list, output_query=str)
    def run(self, output: list, output_query: str):
        return {"output": output, "output_query": output_query}


def generation_pipeline(db_path) -> Pipeline:
    pipeline = Pipeline()
    pipeline.max_loops_allowed = 10

    sql_executor = SQLExecutorComponent(SQLExecutor(db_path))

    agent = SQLAgent()
    pipeline.add_component("agent", agent)
    pipeline.add_component("sql_executor", sql_executor)
    pipeline.add_component("output", OutputSink())
    pipeline.connect("agent.query", "sql_executor.query")
    pipeline.connect("sql_executor.result", "agent.result")
    pipeline.connect("agent.result", "output.output")
    pipeline.connect("agent.result_query", "output.output_query")

    return pipeline


class AgentSQLGenerator:
    def __init__(self, pipeline) -> None:
        self.pipeline = pipeline

    def generate(self, question: str):
        try:
            result = self.pipeline.run({"agent": {"question": question}})
            return result["output"]["output_query"]
        except Exception as e:
            logger.warning("".join(traceback.format_tb(tb=e.__traceback__)))
            return EMPTY_QUERY


def load_agent(db_path):
    pipeline = generation_pipeline(Path(db_path))
    return AgentSQLGenerator(pipeline)


if __name__ == "__main__":
    pipeline = generation_pipeline(DB_FILE)
    logger.info(
        pipeline.run(
            {
                "agent": {
                    "question": "What's the percentage distribution of responses on how much time the Individual Contributors spend answering questions?"
                }
            }
        )
    )
