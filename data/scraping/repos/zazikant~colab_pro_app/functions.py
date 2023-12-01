from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import pandas as pd
from pandasai.llm.openai import OpenAI
import re

import requests
import csv

import matplotlib.pyplot as plt
import io

# Remove duplicate load_dotenv() calls
load_dotenv(find_dotenv())
load_dotenv()

from pandasai import SmartDataframe
from pandasai.llm import OpenAI

# Remove duplicate find_dotenv(), load_dotenv() calls
from dotenv import find_dotenv, load_dotenv

from langchain.document_loaders import PyPDFLoader

import os
import openai
from langchain.llms import OpenAI
from langchain.output_parsers import ResponseSchema
from langchain.output_parsers import StructuredOutputParser
from langchain.memory import ConversationSummaryBufferMemory

from langchain.chains.summarize import load_summarize_chain

from langchain.document_loaders import DirectoryLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables from the .env file
load_dotenv()

from langchain.prompts import ChatPromptTemplate
from langchain import PromptTemplate, LLMChain

def parser(text):
 
    llm = OpenAI()

    context = text.strip()

    email_schema = ResponseSchema(
        name="email_parser",
        description="extract the email id from the text. If required, strip and correct it in format like sample@xyz.com. Only provide these words. If no email id is present, return null@null.com",
    )
    subject_schema = ResponseSchema(
        name="content", description="Just extract the key content by removing email ids and also trimming text related to email ids. Do not add any interpretation."
    )

    response_schemas = [email_schema, subject_schema]

    parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = parser.get_format_instructions()

    template = """
    Interprete the text and evaluate the text.
    email_parser: extract the email id from the text. Only provide these words. If no email id is present, return null@null.com. Use 1 line.
    content: Just extract the content removing email ids. Do not add any interpretation.

    text: {context}

    Just return the JSON, do not add ANYTHING, NO INTERPRETATION!
    {format_instructions}:"""

    #imprtant to have the format instructions in the template represented as {format_instructions}:"""

    #very important to note that the format instructions is the json format that consists of the output key and value pair. It could be multiple key value pairs. All the context with input variables should be written above that in the template.

    prompt  = PromptTemplate(
        input_variables=["context", "format_instructions"],
        template=template
    )

    chain = LLMChain(llm=llm, prompt=prompt, output_key= "testi")
    response = chain.run({"context": context, "format_instructions": format_instructions})

    output_dict = parser.parse(response)
    return output_dict

    
def draft_email(user_input):
    # Define the API endpoint URL and parameters
    url = "http://13.232.224.37:8080/aurum/rest/v1/location/db/findall"
    params = {
        "project_id": 1,
        "user_id": 640,
        "token": "7efbfacb7556e57d0702",
        "page_size": 7
    }
    
    parser_output = parser(user_input)
    
    email = parser_output["email_parser"]
    
    content = parser_output["content"]

    llm = OpenAI()

    # # Make a GET request for each page and extract the desired fields
    locations = []
    for page_num in range(1, 4):
        params["page_num"] = page_num
        response = requests.get(url, params=params)
        data = response.json()
        for record in data["records"]:
            location = {
                "location_id": record["location_id"],
                "location_name": record["location_name"]
            }
            locations.append(location)

    # # Write the locations to a CSV file
    with open("./shashi/locations.csv", "w", newline="") as csvfile:
        fieldnames = ["location_id", "location_name"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for location in locations:
            writer.writerow(location)
            
    df = pd.read_csv("./shashi/locations.csv")

    sdf = SmartDataframe(df, config={"llm": llm})

    sdf.chat(content)      

    response = sdf.last_code_generated.__str__()

    return email, response