from langchain.utilities import SQLDatabase
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationChain, LLMChain
from langchain.memory import ConversationBufferMemory
from template import generateConversationTemplate, generateNaturalResponseTemplate, modifyQuestionTemplate
import re


class DBObject():
    db: SQLDatabase
    conversion_chain: LLMChain
    chat_chain: LLMChain
    response_chain: LLMChain

    def __init__(self, db: SQLDatabase, llm):
        self.db = db

        chat_template = PromptTemplate(template=generateConversationTemplate(), input_variables=["schema", "input"])
        
        response_template = PromptTemplate(
            template=generateNaturalResponseTemplate(),
            input_variables=["input", "data"]
        )
        
        conversion_template = PromptTemplate(template=modifyQuestionTemplate(), input_variables=["schema", "input"])
        
        self.chat_chain = LLMChain(
            prompt=chat_template,
            llm=llm
        )
        
        self.response_chain = LLMChain(
            prompt=response_template,
            llm=llm
        )
        self.conversion_chain = LLMChain(
            prompt=conversion_template,
            llm=llm
        )


    def get_schema(self, hints: list[str]) -> str:
        return self.db.get_table_info_no_throw()

    def make_query(
        self, question: str, hints: list[str]
    ) -> str:
        
        
        schema = self.get_schema(hints)
        
        converted_input = "In this database, " + question
        
        sql_query = self.chat_chain.predict(
            input= converted_input,
            schema=schema
        )
        
        print("QUESTION: "+question)
        print("GENERATED QUERY: "+sql_query.strip())

        response = self.db.run_no_throw(sql_query.strip())

        human_readable = self.response_chain.predict(
            input=converted_input,
            data=response,
        )

        del question
        del response
        del sql_query
        
        return human_readable
