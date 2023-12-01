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


SIMPLE_TEMPLATE = """
Your goal is to use all your knoledge from OpenAI and ChatGPT to interact with the user with simple, no-code anwsers to most of topic questions.

Give greetings to the user and provide answers to questions.
ONLY provide text response to users questions.


Now provide answers:
"""
# Examples of user questions:
# - Hi! how are you?
# - Nice to meet you!
# - Thanks for your help
# - do you know what day is today?
# - do you know what hour is in Buenos Aires, Argentina right now?

class SimpleGeneratorLLM:
    def __init__(self,_snowflake_session: Session, llm: BaseLanguageModel = LLM, simple_template: str = SIMPLE_TEMPLATE):
        self.llm = llm
        self.template = simple_template
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
        # Now format the template with both context and user_input
        final_template = PromptTemplate.from_template(template=self.template)
        
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

  