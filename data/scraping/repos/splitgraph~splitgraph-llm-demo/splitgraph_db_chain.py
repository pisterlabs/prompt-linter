from dataclasses import dataclass
from typing import Any

from langchain import OpenAI, SQLDatabase, SQLDatabaseChain


@dataclass
class LLMQueryResult:
    sql_query: str
    result: Any


def invoke_chain(db: SQLDatabase, query: str, verbose=False) -> LLMQueryResult:
    # Initialize the language model
    llm = OpenAI(temperature=0, verbose=True)  # type: ignore
    # Initiailize the langchain LLM chain, which first tries to determine which
    # table(s) to use, then uses the LLM to generate an SQL query, and finally
    # executes this query against the DDN.
    db_chain = SQLDatabaseChain.from_llm(
        llm, db, verbose=verbose, return_direct=True, return_intermediate_steps=True
    )
    result = db_chain(query)
    return LLMQueryResult(
        sql_query=result["intermediate_steps"][2]["sql_cmd"], result=result["result"]
    )
