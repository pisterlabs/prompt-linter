from typing import Dict, List

from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from pydantic import BaseModel, Field
import logging
from utils.config_loaders import (
    get_config,
    get_secrets
)

config = get_config()
secrets = get_secrets(config)


EXTRACTOR_TEMPLATE = """
Your objective is to extract relevant information from the user's question. Specifically, you should identify the names 
of the schemas and tables associated with each schema, and present them in the requested format. It's important to note 
that you should not seek out additional information beyond what the user has provided. Your primary responsibility 
is to extract information and format it appropriately. find patters and words to understand why schemas and tables are
related each other.

ALWAYS USE THE FORMAT INSTRUCTIONS PROVIDED. Failure to do so will result in a failed task.
Only extract the schemas and tables that are explicitly mentioned in the user's question. Do not extract any additional
information beyond what the user has provided.

{format_instructions}
User question : {query}
"""


class Schema(BaseModel):
    schemas: Dict[str, List[str]] = Field(
        description="""
If you haven't explicitly defined the schemas, or if you want to utilize all the tables, 
it is mandatory to substitute the schema names with 'specify_a_table_name'. Additionally, 
please ensure that each schema is accurately associated with its respective list of tables.
        """
    )


class InfoExtractor(BaseModel):
    """
    Info extractor model
    """
    query: str = Field(..., description="Relevant information from the user's question")
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

        parser = PydanticOutputParser(pydantic_object=Schema)

        prompt = ChatPromptTemplate(
            messages=[
                HumanMessagePromptTemplate.from_template(EXTRACTOR_TEMPLATE)
            ],
            input_variables=["query"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            }
        )

        try:
            logging.info('Extracting information from the user\'s question...')
            _input = prompt.format_prompt(query=self.query)

            output = model(_input.to_messages())

            result = parser.parse(output.content).dict()
            logging.info(f'Successfully extracted information from the user\'s question: {result}')
            return result
        except Exception as e:
            logging.error(f'Error extracting information from the user\'s question: {e}')
            raise e
