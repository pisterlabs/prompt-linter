from pathlib import Path
from textwrap import dedent
from typing import Union

import langchain
import pandas as pd
from langchain.cache import SQLiteCache
from langchain.chat_models import ChatOpenAI
from langchain.prompts import (ChatPromptTemplate, HumanMessagePromptTemplate,
                               SystemMessagePromptTemplate)


def expand_on_results(docs: Union[str, list], user_query: str, sql_results: pd.DataFrame) -> str:
    if isinstance(docs, str):
        docs = [docs]

    chat = ChatOpenAI(temperature=0.0, verbose=True)  # type: ignore

    langchain.llm_cache = SQLiteCache(database_path="cache/.langchain.db")

    template = dedent("""
    Given a user's question and the corresponding SQL query results, use the additional context provided to generate a comprehensive and succinct summary of the query and its results. 
    The additional context should be used specifically to enhance the understanding and interpretation of the user's question and SQL results, rather than being summarized independently.
    
    Extra context:
    {docs}""")

    human_template = dedent("""
    User query: {user_query}
    SQL results: {sql_results}""")

    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    return chat(
        chat_prompt.format_prompt(
            docs="\n\n".join([Path(doc).read_text() for doc in docs]),
            user_query=user_query,
            sql_results=sql_results.to_dict(orient="records"),
        ).to_messages()
    ).content
