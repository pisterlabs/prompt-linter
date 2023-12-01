from langchain.tools import BaseTool, Tool, tool
from pydantic import BaseModel, Field
from langchain.agents import create_csv_agent, AgentType, initialize_agent

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.schema import BaseOutputParser
from langchain.prompts import FewShotChatMessagePromptTemplate, PromptTemplate
from langchain.prompts import FewShotChatMessagePromptTemplate
# Import Azure OpenAI
from langchain.llms import AzureOpenAI
from langchain.chains import LLMChain
from langchain.prompts import FewShotChatMessagePromptTemplate
import pandas as pd
import os
import chainlit as cl
# from chainlit import ask_for_input


from dotenv import load_dotenv
load_dotenv("../../.env")

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain.chat_models import AzureChatOpenAI


class IntentOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""

    def parse(self, text: str):
        """Parse the output of an LLM call."""
        intents = ["GET PARKING SPOT", "GET REGISTRATION DETAILS"]
        for i in intents:
            if i in text:
                return i
        return "INTENT UNCLEAR"


llm = AzureChatOpenAI(
    deployment_name="completion-gpt-35-turbo",
    model_name="gpt-35-turbo",
    temperature=0.0
)

def intent_recognizer_api(query: str) -> str:
    """Identifies the intent of the user's query"""
    intents = ["GET PARKING SPOT", "GET REGISTRATION DETAILS"]
    prompt_template = PromptTemplate.from_template("""You are a helpful assistant that identifies the intent of a user's query. 
    There are 2 possible intents - GET PARKING SPOT, GET REGISTRATION DETAILS. 
    Choose the most appropriate intent based on the human's query.
    What is the intent for the following ask? 
    {ask}
    """)

    intent_chain = LLMChain(llm=llm, prompt=prompt_template, output_key="intent", output_parser=IntentOutputParser(), verbose=True)
    ans = intent_chain.run(query)
    return ans

def get_vehicle_details_string(row):
    values = list(row)
    indices = row.index
    st = ""
    for a,b in list(zip(indices, values)):
        st += a + "-->" + b
        st += ", "
    return st

def fetch_vehicle_details_from_live_database(user):
    df_vehicle_details = pd.read_csv("./data/VehicleDetails.csv")
    df_user_vehicle = df_vehicle_details[df_vehicle_details["Employee"] == user]
    if len(df_user_vehicle) == 0:
        "NO REGISTERED VEHICLE FOUND"
    else:
        # return df_user_vehicle.iloc[0,:]["Outlook/Building Name"]
        return get_vehicle_details_string(df_user_vehicle.iloc[0,:])

def get_employee_building_name_api(user: str) -> str:
    """Returns the building name for an employee"""
    ans = fetch_vehicle_details_from_live_database(user)
    return ans

def fetch_parking_spots_in_building_api(building_name: str) -> str:
    """Performs a lookup on the database to identify the available parking spots in a specific building"""
    # ans = fetch_available_parking_spots_in_building(building_name)
    agent = create_csv_agent(
        llm,
        "./data/AvailableSpace.csv",
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    )
    ans = agent.run(f"Get me all the available parking spots in building {building_name}")
    return ans

async def ask_user_for_clarification_api(query_for_user: str) -> str:
    """Asks the user for any required information/input."""
    print(query_for_user)
    input_from_user = input()
    return input_from_user


def get_parking_reservation_agent():
    t1 = Tool.from_function(
            func=intent_recognizer_api,
            name="get_intent_for_the_users_ask",
            description="Identifies the intent for the users ask"
    )

    t2 = Tool.from_function(
            func=get_employee_building_name_api,
            name="get_employee_building_name",
            description="Returns the building name for an employee"
    )

    t3 = Tool.from_function(
            func=fetch_parking_spots_in_building_api,
            name="fetch_available_parking_spots_in_building",
            description="Performs a lookup on the database to identify the available parking spots in a specific building"
    )

    t4 = Tool.from_function(
            func=ask_user_for_clarification_api,
            name="ask_user_for_clarification",
            description="Asks the user for any required information/input."
    )

    agent = initialize_agent(
            [t2, t3, t4], 
        llm, 
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
        verbose=True)
    return agent