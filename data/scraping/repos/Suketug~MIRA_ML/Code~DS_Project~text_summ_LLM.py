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

# Prompt Template
prompt_text = """
You are a highly specialized AI trained to understand mortgage guidelines, terms, and policies. Your task is to read the following mortgage-related text and summarize it in a way that retains all the essential information but is concise and easy to understand.

Original Text: {input}

Please provide a summarized version of this text.
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

# Output Parser
class CustomOutputParser(AgentOutputParser):

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

output_parser = CustomOutputParser()

# Set up LLM
llm = ChatOpenAI(model_name="gpt-3.5-turbo-0613", temperature=0.7)

# LLM Chain
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Set up the Agent
agent = LLMSingleActionAgent(
    llm_chain=llm_chain,
    output_parser=output_parser,
    stop=["\n"]  # You can adjust this as needed
)

# Agent Executor
agent_executor = AgentExecutor(agent=agent, tools=[])

# Initialize a DataFrame to store the summarized text, if 'summarized_text.csv' exists, it will load it
try:
    df_summarized = pd.read_csv("Data/Extracted_Data/summarized_text.csv")
except FileNotFoundError:
    df_summarized = pd.DataFrame()

# Read the input CSV (you can loop through your broken CSV files)
input_csv_path = "Data/Extracted_Data/CSV's/Clean_Master_data1_part_9.csv"
df = pd.read_csv(input_csv_path)

# Initialize an empty list to store the summarized text
summarized_texts = []

# Iterate through the DataFrame
for index, row in df.iterrows():
    detailed_text = row['Description']
    raw_response = agent_executor.run({"input": detailed_text})
    # raw_response is already the summarized text
    summarized = raw_response
    summarized_texts.append(summarized)

# Add the summarized text to the DataFrame
df['summarized_text'] = summarized_texts

# Append to the existing DataFrame for summarized text
df_summarized = pd.concat([df_summarized, df], ignore_index=True)

# Save the updated DataFrame back to the same CSV file
df_summarized.to_csv("Data/Extracted_Data/summarized_text.csv", index=False)

