# Source: langchain.SQLDatabaseChain
"""Chain for interacting with SQL Database."""
from typing import Dict, List, Any

from pydantic import BaseModel, Extra, Field

import json

from langchain.chains.base import Chain
from langchain.chains.llm import LLMChain
from langchain.input import print_text
from langchain.llms.base import LLM

from backend.modules.database.prompt import PROMPT
from backend.modules.database.sql_database import SQLDatabase
from backend.modules.database.encoder import UUIDEncoder


class SQLDatabaseChain(Chain, BaseModel):
    """Chain for interacting with SQL Database.

    Example:
        .. code-block:: python

            from langchain import SQLDatabaseChain, OpenAI, SQLDatabase
            db = SQLDatabase(...)
            db_chain = SelfAskWithSearchChain(llm=OpenAI(), database=db)
    """

    llm: LLM
    """LLM wrapper to use."""
    database: Any  # SQLDatabase
    """SQL Database to connect to."""
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:
    debug: bool = False  # even more verbose
    custom_memory: dict = {}

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Return the singular input key.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the singular output key.

        :meta private:
        """
        return [self.output_key]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        llm_chain = LLMChain(llm=self.llm, prompt=PROMPT, verbose=self.debug)
        input_text = f"{inputs[self.input_key]} \nSQLQuery:"
        llm_inputs = {
            "input": input_text,
            "dialect": self.database.dialect,
            "table_info": self.database.table_info,
            "stop": ["\nSQLResult:"],
        }
        sql_cmd = llm_chain.predict(**llm_inputs)
        if self.verbose:
            print_text(input_text)
            print_text(sql_cmd, color="green")
        error, result = self.database.run(sql_cmd)
        lang_output = ""
        if error:
            lang_output = "Error: " + error
        else:
            lang_output = f"{len(result)} rows returned:"
            if len(result) <= 3:
                lang_output += "\n" + "\n".join(map(str, result))
            else:
                lang_output += f"\n[{result[0]}, ...]"
                lang_output += f"\nAll results are stored in query_result.json"
                with open("query_result.json", "w") as f:
                    json.dump(result, f, cls=UUIDEncoder, default=str)

        if self.verbose:
            print_text("\nSQLResult: ")
            print_text(lang_output, color="yellow")
            print_text("\nAnswer:")
        input_text += f"{sql_cmd}\nSQLResult: {lang_output}\nAnswer:"
        llm_inputs["input"] = input_text
        final_result = llm_chain.predict(**llm_inputs)
        if self.verbose:
            print_text(final_result, color="green")
        self.custom_memory = {
            "query_result": result,
            "sql_cmd": sql_cmd,
            "query_result_summary": {
                "outline": f"{len(result)} rows",
                "source": sql_cmd,
                "text": lang_output,
            }
        }
        if len(result) > 3:
            self.custom_memory["query_result_summary"] = {
                **self.custom_memory["query_result_summary"],
                "type": "json file",
                "name": "query_result.json",
                "example": f"[{result[0]}, ...]",
            }

        return {self.output_key: final_result}
