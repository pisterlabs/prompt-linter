from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field
from typing import Dict
import logging
from utils.config_loaders import (
    get_config,
    get_secrets
)

config = get_config()
secrets = get_secrets(config)

EXTRACTOR_TEMPLATE = """
Given an input question, first create a syntactically correct {dialect} query to run, 
then look at the results of the query and return the answer. Unless the user specifies in his question a specific 
number of examples he wishes to obtain, always limit your query to at most {top_k} results. 
You can order the results by a relevant column to return the most interesting examples in the database.

Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema description. Be careful to not query for 
columns that do not exist. Also, pay attention to which column is in which table.

Always try to use the primary keys and the foreign keys to join the tables.

Only use the tables listed below.

{table_info}

User Question: {input}

ALWAYS USE THE FORMAT INSTRUCTIONS PROVIDED. Failure to do so will result in a failed task.

{format_instructions}

Give me the query that answers the question above using the tables listed above.
"""


class Query(BaseModel):
    # create a function to generate uuid
    query: str = Field(
        description="""
        The query that answers the question. Always try to use the primary keys and the foreign keys to join the tables.
        """
    )


class SQLRunner(BaseModel):
    """
    Info extractor model
    """
    input: str = Field(..., description="User's question")
    table_info: Dict = Field(..., description="Table info contains the metadata of the tables related to the question")
    dialect: str = Field(..., description="Dialect of the database, PostgreSQL, MySQL, etc.")
    top_k: int = Field(30, description="Number of results to return, rows of the query")
    model_name: str = Field('gpt-3.5-turbo', description="Name of the model to use.")
    temperature: int = Field(0, description="Temperature of the model to use.")

    def get_result(self) -> Dict:
        """
        Get result of the info extractor
        Returns: result of the info extractor a dict that contains the schemas and tables
        for example: {"schemas": {"schema1": ["table1", "table2"], "schema2": ["table3", "table4"]}}
        """
        model = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            openai_api_key=secrets['openai_api']['token'],
            openai_organization=secrets['openai_api']['organization']
        )

        parser = PydanticOutputParser(pydantic_object=Query)

        prompt = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template(EXTRACTOR_TEMPLATE)
            ],
            input_variables=["input", "table_info", "dialect", "top_k"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            }
        )

        try:
            logging.info('trying to get the sql query')
            _input = prompt.format_prompt(
                input=self.input,
                table_info=self.table_info,
                dialect=self.dialect,
                top_k=self.top_k
            )

            output = model(_input.to_messages())

            result = parser.parse(output.content).dict()

            return result
        except Exception as e:
            logging.error(f'Error extracting information from the user\'s question: {e}')
            raise e
