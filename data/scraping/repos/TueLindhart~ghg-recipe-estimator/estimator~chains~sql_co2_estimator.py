import os
from typing import Literal

from langchain.chat_models import ChatOpenAI
from langchain.sql_database import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain

from estimator.prompt_templates.sql_co2_estimator import (
    DK_CO2_SQL_PROMPT_TEMPLATE,
    EN_CO2_SQL_PROMPT_TEMPLATE,
)


def get_co2_sql_chain(language: Literal["da", "en"], verbose: bool = False):
    sql_dk_co2_db = SQLDatabase.from_uri(
        f"sqlite:///{os.getcwd()}/estimator/data/dk_co2_emission.db", sample_rows_in_table_info=2
    )
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k")  # type: ignore
    co2_sql_chain = SQLDatabaseChain.from_llm(
        llm=llm,
        db=sql_dk_co2_db,
        verbose=verbose,
        prompt=EN_CO2_SQL_PROMPT_TEMPLATE if language == "en" else DK_CO2_SQL_PROMPT_TEMPLATE,
        top_k=200,
    )

    return co2_sql_chain
