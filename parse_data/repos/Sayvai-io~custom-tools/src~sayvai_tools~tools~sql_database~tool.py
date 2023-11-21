from typing import Optional

from langchain.schema import BasePromptTemplate
from langchain.schema.language_model import BaseLanguageModel
from sqlalchemy.engine import Engine

from sayvai_tools.tools.sql_database.prompt import PROMPT, SQL_PROMPTS
from sayvai_tools.utils.database.dbbase import SQLDatabase
from sayvai_tools.utils.database.dbchain import SQLDatabaseChain


class Database:
    """Tool that queries vector database."""

    name = "Database"
    description = (
        "Useful for when you need to access sql database"
        "Input should be a natural language"
    )

    def __init__(
        self,
        llm: BaseLanguageModel,
        engine: Engine,
        prompt: Optional[BasePromptTemplate] = None,
        verbose: bool = False,
        k: int = 5,
    ):
        self.llm = llm
        self.engine = engine
        self.prompt = prompt
        self.verbose = verbose
        self.k = k

    def _run(self, query: str) -> str:
        db = SQLDatabase(engine=self.engine)

        if self.prompt is not None:
            prompt_to_use = self.prompt
        elif db.dialect in SQL_PROMPTS:
            prompt_to_use = SQL_PROMPTS[db.dialect]
        else:
            prompt_to_use = PROMPT
        inputs = {
            "input": lambda x: x["question"] + "\nSQLQuery: ",
            "top_k": lambda _: self.k,
            "table_info": lambda x: db.get_table_info(
                table_names=x.get("table_names_to_use")
            ),
        }
        if "dialect" in prompt_to_use.input_variables:
            inputs["dialect"] = lambda _: (db.dialect, prompt_to_use)

        sql_db_chain = SQLDatabaseChain.from_llm(
            llm=self.llm, db=db, prompt=prompt_to_use, verbose=self.verbose
        )

        return sql_db_chain.run(query)

    async def _arun(self, query: str):

        raise NotImplementedError("SQL database async not implemented")
