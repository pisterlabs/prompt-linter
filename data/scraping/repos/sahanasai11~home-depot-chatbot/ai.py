from langchain.agents import Tool
from langchain.agents import initialize_agent
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

import os
from pathlib import Path
from dotenv import load_dotenv

PROMPT_TEMPLATE = """
You are a customer service worker for the company Home Depot that answers questions about Home
Depot products. You have access to the following SQL table that describes 
products at Home Depot which you will use to answer as if you are a human:

Table 1: home_depot_data_2021 with columns: url, name, description, brand, price, currency, breadcrumbs, overview, specifications.
The url column indicates the url for the product in the online store.
The name column indicates the name of the product.
The description column provides the description of the product.
The brand column provides the brand of the product.
The price column provides the price of the product.
The currency column indicates the currency at which the price of the product is listed at.
The breadcrumbs column indicates the type of product or appliance the product is.
The overview provides an overview of the product.
The specifications provides specific details about the product.

When you are asked to retrieve data, use a SQL query using the provided table
and return an answer as if you are a human. If the answer's content is too long, 
provide a one sentence of the first result.
###
{query}
"""

# Table 1: home_depot_data_2023 with columns: details, seller, title, and url
# The details column provides a description of the product
# The seller column provides the seller or brand of the product
# The title column provides the name of the product
# The url column provides the url to the product on the Home Depot website

env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# class to set LLM properties and initialize agent
class AI:
    def __init__(self, sql_query):
        self.llm = OpenAI(temperature=0.75, openai_api_key=os.environ['OPEN_AI_API_KEY'])
        self.prompt = PromptTemplate(
            input_variables=["query"],
            template=PROMPT_TEMPLATE,
        )
        self.tools = [
            Tool(
                name = "SQL",
                func=sql_query,
                description="Runs a given SQL query and returns response as Markdown"
            )
        ]
        self.agent = initialize_agent(
            self.tools, 
            self.llm,
            agent="zero-shot-react-description", 
            verbose=True, 
            max_iterations=4,
            max_execution_time=3,
            early_stopping_method="generate"
        )

    def run(self, query):
        agent_prompt = self.prompt.format(query=query)
        return self.agent.run(agent_prompt)
