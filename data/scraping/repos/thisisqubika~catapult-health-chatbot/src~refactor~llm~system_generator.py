from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema.language_model import BaseLanguageModel
import streamlit as st
from snowflake.snowpark import Session
import os
from langchain_experimental.sql import SQLDatabaseChain
from config import db, OPENAI_API_KEY
from llm.streamming_handler import StreamHandler


LLM = ChatOpenAI(
    temperature=0.0,
    model="gpt-4-1106-preview",
    streaming=True,
    openai_api_key =OPENAI_API_KEY,
)


QUALIFIED_TABLE_NAME = "CATAPULT_HEALTH_DB.POC_CATAPULT_HEALTH.HEALTHRECORDDATA"

TABLE_DESCRIPTION = """
This table is an electronic health record (EHR) system or a patient health database which has clinical or healthcare data from patients.
"""

METADATA_QUERY = "SELECT VARIABLE_NAME, DEFINITION FROM CATAPULT_HEALTH_DB.POC_CATAPULT_HEALTH.HEALTHRECORDDATA_ATTRIBUTES;"


SYSTEM_TEMPLATE = """
Your goal is to give correct, executable Snowflake sql query to users.
You will be acting as an AI Snowflake SQL Expert named Catapult Health Bot.
You will be replying to users who will be confused if you don't respond in the character of Catapult Health Bot.
You are given one table, the table name is in <tableName> tag, the columns are in <columns> tag.
You should check the table columns,data types and metadata to provide sample questions based on the table schema and available columns.

{context}

```
Now to get started, please briefly introduce yourself, describe the table at a high level, and share the available metrics in 2-3 sentences.
Then provide 3 example questions using bullet points, but DO NOT provide sql code.
ONLY provide text response

"""


@st.cache_data(show_spinner=False)
def get_table_context(_snowflake_session:Session, table_name: str, table_description: str, metadata_query: str = None):
    table = table_name.split(".")

    # Execute the query
    query = f"""
        SELECT COLUMN_NAME, DATA_TYPE FROM {table[0].upper()}.INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = '{table[1].upper()}' AND TABLE_NAME = '{table[2].upper()}'
    """
    result = _snowflake_session.sql(query).collect()

    # Process the result
    columns = "\n".join(
        [
            f"- **{row['COLUMN_NAME']}**: {row['DATA_TYPE']}"
            for row in result
        ]
    )

    context = f"""
    Here is the table name <tableName> {'.'.join(table)} </tableName>
    <tableDescription>{table_description}</tableDescription>
    Here are the columns of the {'.'.join(table)}
    <columns>\n\n{columns}\n\n</columns>
    """

    if metadata_query:
        metadata_result = _snowflake_session.sql(metadata_query).collect()
        metadata = "\n".join(
            [
                f"- **{row['VARIABLE_NAME']}**: {row['DEFINITION']}"
                for row in metadata_result
            ]
        )
        context += f"\n\nAvailable variables by VARIABLE_NAME:\n\n{metadata}"

    return context




class SystemGeneratorLLM:
    def __init__(self,_snowflake_session: Session, llm: BaseLanguageModel = LLM,system_template: str = SYSTEM_TEMPLATE):
        self.llm = llm
        self.template = system_template
        self.snowflake_session = _snowflake_session
        self.stream_handler = StreamHandler(st.empty())
    
    def _get_connection(self):
        # create connection:
        db_chain = SQLDatabaseChain.from_llm(
            llm=self.llm,
            db=db,
            verbose=True,
            return_direct=True)
        
        return db_chain.run
    
    # PromptTemplate
    def _build_prompt(self) -> str:
        table_context = get_table_context(
            _snowflake_session=self.snowflake_session,
            table_name=QUALIFIED_TABLE_NAME,
            table_description=TABLE_DESCRIPTION,
            metadata_query=METADATA_QUERY)

        # Now format the template with both context and user_input
        formatted_template = self.template.format(context=table_context)

        final_template = PromptTemplate.from_template(template=formatted_template)
        
        return str(final_template)


    def generate_response(self):
        prompt = self._build_prompt()
        response_stream = self.llm.stream(prompt)

        # Clear the existing text in the stream handler
        self.stream_handler.text = ""

        # Stream the response and let the handler update the text
        for chunk in response_stream:
            self.stream_handler.on_llm_new_token(chunk)

        # The final response should now be in self.stream_handler.text
        # Remove any unwanted parts from the response if necessary
        response = self.stream_handler.text.replace("content=", "").strip()

        return response


    def _build_chain(self) -> LLMChain:
        return LLMChain(
            llm=self.llm,
            prompt=self._build_prompt(),  # Pass user input here
            verbose=True
            )

    