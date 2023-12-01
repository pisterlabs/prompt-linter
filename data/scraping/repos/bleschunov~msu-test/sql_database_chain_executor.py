import dataclasses

import langchain
import pandas as pd
from dotenv import load_dotenv
from langchain.callbacks import StdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from openai import OpenAIError
from sqlalchemy.exc import SQLAlchemyError

from config.config import config
from datastep.components.chain import get_sql_database_chain_patched
from datastep.components.datastep_prediction import DatastepPrediction
from datastep.components.datastep_prompt import DatastepPrompt
from datastep.components.patched_database_class import SQLDatabasePatched
from datastep.components.patched_sql_chain import SQLDatabaseChainPatched
from datastep.models.intermediate_steps import IntermediateSteps
from repository.prompt_repository import prompt_repository

load_dotenv()


TABLE_PROMPT_ID_MAPPING = {
    "платежи": 1,
    "сотрудники": 2
}


@dataclasses.dataclass
class SQLDatabaseChainExecutor:
    db_chain: SQLDatabaseChainPatched
    debug: bool = False
    verbose_answer: bool = False
    langchain_debug: bool = False

    def __post_init__(self):
        langchain.debug = self.langchain_debug

    def run(self, query: str, tables: list[str]) -> DatastepPrediction:
        callbacks = self.get_callbacks()

        if tables:
            self.db_chain.database._include_tables = set(tables)
            prompt_dto = prompt_repository.fetch_by_id(TABLE_PROMPT_ID_MAPPING[tables[0]])
        else:
            prompt_dto = prompt_repository.fetch_by_id(config["prompt_id"])

        self.db_chain.llm_chain.prompt = DatastepPrompt.get_prompt(table_description=prompt_dto.prompt)

        try:
            db_chain_response = self.db_chain(query, callbacks=callbacks)
        except OpenAIError as e:
            return DatastepPrediction(
                answer=str(e),
                sql=None,
                table=None,
                table_source=None,
                is_exception=True
            )
        except SQLAlchemyError as e:
            intermediate_steps = e.intermediate_steps
            intermediate_steps = IntermediateSteps.from_chain_steps(intermediate_steps)

            return DatastepPrediction(
                answer=str(e),
                sql=self.get_sql_markdown(intermediate_steps.sql_query),
                table=None,
                table_source=None,
                is_exception=True
            )

        if self.verbose_answer:
            chain_answer = db_chain_response["result"]
        else:
            chain_answer = ""

        intermediate_steps = db_chain_response["intermediate_steps"]
        intermediate_steps = IntermediateSteps.from_chain_steps(intermediate_steps)
        sql_query = intermediate_steps.sql_query
        sql_result = intermediate_steps.sql_result

        return DatastepPrediction(
            answer=chain_answer,
            sql=self.get_sql_markdown(sql_query),
            table=self.get_table_markdown(sql_result),
            table_source=self.get_table_source(sql_result),
            is_exception=False
        )

    @classmethod
    def get_sql_markdown(cls, sql_result) -> str:
        if sql_result:
            return f"~~~sql\n{sql_result}\n~~~"
        return ""

    @classmethod
    def get_table_markdown(cls, sql_result) -> str:
        data_frame = pd.DataFrame(sql_result)

        if data_frame is not None and any(data_frame):
            return data_frame.to_markdown(index=False, floatfmt=".3f")

        return ""

    @classmethod
    def get_table_source(cls, sql_result) -> str:
        return pd.DataFrame(sql_result).to_json(orient="table", force_ascii=False, index=False)

    def get_callbacks(self):
        callbacks = []
        if self.debug:
            callbacks.append(StdOutCallbackHandler())

        return callbacks


def get_sql_database_chain_executor(
    db: SQLDatabasePatched,
    llm: ChatOpenAI,
    debug: bool = False,
    verbose_answer: bool = False
) -> SQLDatabaseChainExecutor:
    return SQLDatabaseChainExecutor(
        db_chain=get_sql_database_chain_patched(db, llm, DatastepPrompt.get_prompt(), verbose_answer),
        debug=debug,
        verbose_answer=verbose_answer
    )
