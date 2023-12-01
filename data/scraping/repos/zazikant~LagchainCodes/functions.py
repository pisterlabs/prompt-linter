from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from dotenv import find_dotenv, load_dotenv
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

import os
import openai
import pprint
import json
import pandas as pd
from pandasai.llm.openai import OpenAI
from dotenv import load_dotenv

import requests
import csv

import matplotlib.pyplot as plt
import io

load_dotenv(find_dotenv())

load_dotenv()

from dotenv import find_dotenv, load_dotenv

import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm import OpenAI

    
def draft_email(user_input):
    # Define the API endpoint URL and parameters
    url = "http://13.232.224.37:8080/aurum/rest/v1/location/db/findall"
    params = {
        "project_id": 1,
        "user_id": 640,
        "token": "7efbfacb7556e57d0702",
        "page_size": 2
    }

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

    sdf.chat(user_input)        

    response = sdf.last_code_generated.__str__()

    return response