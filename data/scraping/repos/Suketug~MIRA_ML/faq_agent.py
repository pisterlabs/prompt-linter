from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import StringPromptTemplate
from langchain import OpenAI, LLMChain
from langchain.chat_models import ChatOpenAI
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish, OutputParserException
import pandas as pd
import json
import os
import openai
import dotenv
from pydantic import Field

# Load the .env file
dotenv.load_dotenv()

# Initialize OpenAI
API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = API_KEY

# Prompt Template for FAQ generation
prompt_text = """
You are a highly specialized AI trained to understand mortgage guidelines, terms, and policies. Your task is to generate Frequently Asked Questions (FAQs) based on the provided information.

summarized_text: {input}

Please generate a relevant FAQ question.
"""

class CustomPromptTemplate(StringPromptTemplate):
    template: str = Field(..., description="The template text for the prompt")

    def __init__(self, template, input_variables, tools=[]):
        super().__init__(input_variables=input_variables, tools=tools)
        self.template = template  # Initialize the instance attribute

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

# Create an instance of CustomPromptTemplate
prompt = CustomPromptTemplate(template=prompt_text, input_variables=["input"], tools=[])

# Output Parser for FAQ questions
class FAQOutputParser(AgentOutputParser):
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        summarized = llm_output.strip()
        # Return AgentFinish object
        return AgentFinish(
            return_values={
                "summarized_text": summarized,
                "output": llm_output
            },
            log={"llm_output": llm_output}  
        )

faq_output_parser = FAQOutputParser()

# Set up LLM for FAQ generation
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0.7)

# LLM Chain
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Set up the Agent for FAQ generation
faq_agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=faq_output_parser,
    stop=["\n"]  # You can adjust this as needed
)

# Agent Executor
faq_agent_executor = AgentExecutor(agent=faq_agent, tools=[])

# Initialize a DataFrame to store the summarized text, if 'summarized_text.csv' exists, it will load it
try:
    df_faq = pd.read_csv("Data/Extracted_Data/master_with_faq.csv")
except FileNotFoundError:
    df_faq = pd.DataFrame()

# Read your input CSV file
input_csv_path = "Data/Extracted_Data/FAQ_CSV's/master_without_faq5.csv"
df = pd.read_csv(input_csv_path)

# Initialize an empty list to store generated FAQ questions
faq_questions = []

# Generate FAQ questions based on the `summarized_text` column of the DataFrame
for index, row in df.iterrows():
    summarized_text = row['summarized_text']
    faq_response = faq_agent_executor.run({"input": summarized_text})
    faq_question = faq_response.strip()
    faq_questions.append(faq_question)

# Add the FAQ questions to the DataFrame
df['FAQ'] = faq_questions

# Append to the existing DataFrame for summarized text
df_faq = pd.concat([df_faq, df], ignore_index=True)

# Save the updated DataFrame with FAQ questions back to a new CSV file
df_faq.to_csv("Data/Extracted_Data/master_with_faq.csv", index=False)
