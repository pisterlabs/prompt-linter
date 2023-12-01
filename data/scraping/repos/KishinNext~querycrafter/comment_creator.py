import logging
from typing import Dict, List

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from pandas import DataFrame
from pydantic import BaseModel, Field
from pydantic import BaseModel as PydanticBaseModel

from utils.config_loaders import (
    get_config,
    get_secrets
)

config = get_config()
secrets = get_secrets(config)

EXTRACTOR_TEMPLATE = """
Your task is to write comments explaining the meaning of each column in the specified tables. Enhance, correct, 
and provide coherent comments based on the metadata information of the tables and the data they contain. Do not request 
additional information. Your sole task is to create comments according to the requested format.

Schema Name: {schema_name}
Table Name: {table_name}

Table metadata:
{table_info}

Table data:
{table_data}

ALWAYS USE THE FORMAT INSTRUCTIONS PROVIDED. Failure to do so will result in a failed task.
Do not extract any additional information beyond what the user has provided.

{format_instructions}

The JSON must contains the following attributes:
table_name: Name of the table to create the comments.
schema_name: Name of the schema of the table to create the comments.
columns: List of dictionaries, each dictionary contains the column name and the comment associated to the column.

Give me the comments of the columns of the table {table_name} of the schema {schema_name}.
"""


class Table(BaseModel):
    table_name: str = Field(
        description="""
The name of the table, this only include the name of the table, for example 'users', not add the schema name 'public.users'
"""
    )
    schema_name: str = Field(
        description="""
The name of the schema, this only include the name of the schema of the table, for example 'public', not add the table name 'public.users'
"""
    )
    columns: List[Dict] = Field(
        description="""
List of dictionaries, each dictionary contains the column name and the comment associated to the column, 
for example [{'column_name': 'id', 'comment': 'Unique identifier of the user'}]. Never add additional fields to the dictionary like 'type' or 'constraints'
"""
    )


class BaseModel(PydanticBaseModel):
    class Config:
        arbitrary_types_allowed = True


class CommentCreator(BaseModel):
    """
    Column extractor model class. This class is used to validate the input of the model.
    """
    table_name: str = Field(..., description="Name of the table to create the comments")
    schema_name: str = Field(..., description="Name of the schema of the table to create the comments")
    table_info: list = Field(..., description="Metadata information of the tables")
    table_data: DataFrame = Field(..., description="DataFrame that contains the data about the table")
    model_name: str = Field('gpt-3.5-turbo', description="Name of the model to use.")
    temperature: int = Field(0, description="Temperature of the model to use.")

    def get_result(self) -> Dict:
        """
        This method is used to get the result of the model. This return the comments of the columns of one table each
        time.
        Returns: A dictionary with the name of the table, the schema name and the comments of the columns of the table.

        {
            'table_name': 'example',
            'schema_name': 'public',
            'columns': [
                {
                    'column_name': 'has_pregnant_traveler',
                    'comment': 'Boolean value indicating whether the traveler has a pregnancy or not.'
                }
            ]
        }

        """
        model = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            openai_api_key=secrets['openai_api']['token'],
            openai_organization=secrets['openai_api']['organization']
        )

        parser = PydanticOutputParser(pydantic_object=Table)

        prompt = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template(EXTRACTOR_TEMPLATE)
            ],
            input_variables=["table_info", "table_data", "table_name", "schema_name"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            }
        )

        try:
            logging.info('Trying to generate the comments of the columns of the table')
            _input = prompt.format_prompt(table_info=self.table_info, table_data=self.table_data,
                                          table_name=self.table_name, schema_name=self.schema_name)

            output = model(_input.to_messages())

            result = parser.parse(output.content).dict()

            return result
        except Exception as e:
            logging.error(f'Error generating the comments of the columns of the table: {e}')
            raise e
