"""
File that contains the logic for keyword generation.
"""
import logging
from typing import List, Literal
import json

from langchain.agents import AgentType, initialize_agent, AgentExecutor
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate
)
from langchain.agents import Tool
from langchain import SerpAPIWrapper
from langchain.prompts.example_selector import SemanticSimilarityExampleSelector
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import FewShotPromptTemplate, PromptTemplate

from src.logic.langchain_tools.tool_process_thought import process_thoughts

from src.logic.config import secrets as config_secrets

from db.database_connector import DatabaseConnector

class KeywordGenerator():
    """
    Class that contains the logic for keyword generation.
    """

    def __init__(self) -> None:
        self.keyword_list: List[str] = []
        self.few_shot_examples: List = []
        self.template: str = """template"""
        self.system_template: Literal = """system message template"""
        self.system_message_prompt: SystemMessagePromptTemplate = SystemMessagePromptTemplate.from_template(self.system_template)
        self.human_template: Literal = """human message template"""
        self.few_shot_human: SystemMessagePromptTemplate = SystemMessagePromptTemplate.from_template("human message example template", additional_kwargs={"name": "example_user"})
        self.few_shot_ai: SystemMessagePromptTemplate = SystemMessagePromptTemplate.from_template("""ai message example template""", additional_kwargs={"name": "example_assistant"})
        self.human_message_prompt: HumanMessagePromptTemplate = HumanMessagePromptTemplate.from_template(self.human_template)
        self.chat_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
            [self.system_message_prompt, self.few_shot_human, self.few_shot_ai, self.human_message_prompt]
        )
        self.llm: ChatOpenAI = ChatOpenAI(
            model="gpt-4",
            temperature = 0,
            client = self.chat_prompt,
            openai_api_key = config_secrets.read_openai_credentials()
        )
        self.search: SerpAPIWrapper = SerpAPIWrapper(serpapi_api_key = config_secrets.read_serpapi_credentials())
        self.tools = [
            Tool(
                name = "Search",
                func = self.search.run,
                description = "useful for when you need to answer questions about current events"
            ),
            Tool(
                name = "Thought Processing",
                func = process_thoughts,
                description = """useful for when you have a thought that you want to use in a task,
                but you want to make sure it's formatted correctly"""
            )
        ]
        self.agent: AgentExecutor = initialize_agent(
            tools = self.tools, llm = self.llm, agent = AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose = True
        )

    def createDBConnection(self, db_name: str = "risk.db") -> DatabaseConnector:
        """
        Open a connection to the database and return a connection object.
        """
        db = DatabaseConnector(db_name)
        db.open()
        return db
    
    def get_few_shot_examples(self, input: str) -> List[str]:
        """
        Choose example from the database to use for the few-shot learning.

        :param input: The input to the few-shot learning.
        :return: The few-shot learning example.
        """
        with open('./content/examples/keyword_examples.json', 'r') as file:
            data = json.load(file)
        example_prompt: PromptTemplate = PromptTemplate(
            input_variables = ["input", "output"],
            template = "Input: {input}\nOutput: {output}",
        )
        examples = data
        example_selector = SemanticSimilarityExampleSelector.from_examples(
            examples,
            OpenAIEmbeddings(),
            FAISS,
            k = 1,
        )
        similar_prompt: FewShotPromptTemplate = FewShotPromptTemplate(
            example_selector = example_selector,
            example_prompt = example_prompt,
            prefix = "Please identify n keywords for a company.",
            suffix = """{input}\nOutput:""",
            input_variables = ["input"],
        )
        result: str = similar_prompt.format(input = input)
        pairs: List[str] = result.split("\n\n")
        input: str = pairs[1].split("\nOutput:")[0].strip()
        output: str = pairs[1].split("\nOutput:")[1].strip()
        return [input, output]

    def clean_output(self, output_raw: str) -> None:
        """
        Clean the output from the agent.

        :param output_raw: The raw output from the agent.
        :return: The cleaned output.
        """
        if not isinstance(output_raw, str) or not output_raw:
            raise ValueError("Argument output_raw must be a non empty string")
        try:
            keyword_list: List[str] = output_raw.split(", ")
            self.keyword_list = []
            for keyword in keyword_list:
                cleaned_keyword: str = keyword.replace(".", "").replace(" and ", "")
                self.keyword_list.append(cleaned_keyword)
        except ValueError as e:
            logging.error(e)
            raise ValueError(f"Error: {e}") from e

    def generate_keywords(self, company: str, n: int, message_type: str = None) -> List[str]:
        """
        Generate a comma seperated list of keywords that accurately describe a company and its operations.

        :param name: The name of the company for which keywords need to be generated.
        :param num: The number of keywords to generate.
        :return: The resulting list of keywords.
        :raise ValueError: If arg company is not a string or if the string is empty.
        :raise ValueError: If arg num is not a positive integer.
        """
        if not isinstance(company, str) or not company:
            raise ValueError("Argument company must be a non empty string")
        if not isinstance(n, int) or n < 1:
            raise ValueError("Argument num must be a positive integer.")
        try:
            self.template = """As a risk analyst, your task is to generate a comma-separated list of keywords that 
            accurately describe a company based on its operations and products. The company name will be passed to you. Include
            industry-specific terminology to ensure the keywords are tailored to the company's operations. Please return only 
            the comma-separated list of keywords, without any prefix or suffix, containing only the desired amount of keywords 
            as your final answer."""
            self.system_message_prompt = SystemMessagePromptTemplate.from_template(self.template)
            if(message_type and message_type == "keyword"):
                db = self.createDBConnection()
                query = "SELECT message FROM prompts WHERE type = ?"
                result = db.c.execute(query, (message_type,))
                row = result.fetchone()
                if row:
                    self.human_template = row[0]
                    self.few_shot_examples = self.get_few_shot_examples(input=self.human_template)
                else:
                    self.human_template = "Please identify {n} keywords for the company {company}."
                    self.few_shot_examples = self.get_few_shot_examples(input="Please identify {n} keywords for the company {company}.")
            else:
                self.human_template = """Please identify {n} keywords for the company {company}."""
                self.few_shot_examples = self.get_few_shot_examples(input="Please identify {n} keywords for the company {company}.")
            self.few_shot_human = SystemMessagePromptTemplate.from_template(self.few_shot_examples[0], additional_kwargs = {"name": "example_user"})
            self.few_shot_ai = SystemMessagePromptTemplate.from_template(self.few_shot_examples[1], additional_kwargs = {"name": "example_assistant"})
            self.human_message_prompt = HumanMessagePromptTemplate.from_template(self.human_template)
            self.chat_prompt = ChatPromptTemplate.from_messages(
                [self.system_message_prompt, self.few_shot_human, self.few_shot_ai, self.human_message_prompt]
            )
            result: str = self.agent.run(
                self.chat_prompt.format_messages(company = company, n = n)
            )
            self.clean_output(result)
        except ValueError as e:
            if message_type and message_type == "keyword":
                db.close()
            logging.error(e)
            raise ValueError(f"Error: {e}") from e
        if message_type and message_type == "keyword":
            db.close()
        return self.keyword_list
