from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI

import pandas as pd
import requests
import json

def set_up_llm(openai_api_key):
    # Create extraction chain
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0,
        max_tokens=2000,
        openai_api_key=openai_api_key
    )
    return llm

#Setup deal Schema
def extract_deals(llm,scraper_deals):
    if len(scraper_deals) == 0:
        return []
        
    deal_scehema = Object(
        # This what will appear in your output. It's what the fields below will be nested under.
        # It should be the parent of the fields below. Usually it's singular (not plural)
        id="deal",
        
        # Natural language description about your object
        description="Information about a deal",
        
        # Fields you'd like to capture from a piece of text about your object.
        attributes=[
            Text(
                id="name",
                description="The company name."
            ),
            Text(
                id="stage",
                description="The stage of the funding round. Undisclosed if not mentioned."
            ),
            Number(
                id="amount_raised",
                description="The amount of funding raised. MUST be expressed as an INTEGER (NOT STRING) with all included zeros and NO COMMAS or decimals. None if the amount is not mentioned."
            ),
            Text(
                id="current_investors",
                description="The investors in the funding round or None if not mentioned. Multiple investors MUST be separated by semi-colons."
            ),
            Text(
                id="investment_sector",
                description="The sector of the following that best fits the company: Buildings; Energy, Power & Storage; Waste, Plastic & Recycling; Food, Agriculture & Forestry; Industrial Processes & Management; Transportation & Mobility; General Catalyst; Other; Other: CCUS."
            ),
            Text(
                id="geography",
                description="The company location."
            )
        ],
        
        # Examples help go a long way with telling the LLM what you need
        examples=[
            (
                """ 
                ⚡ Low Carbon, a London, UK-based renewables investing and asset management company, raised $388M in debt from ABN AMRO Fund, ING Group, Commonwealth Bank of Australia, and Intesa Sanpaolo.
                """, 
                [
                    {"name": "Low Carbon", "stage": "Undisclosed", "amount_raised": 388000000, "current_investors": "ABN AMRO Fund; ING Group; Commonwealth Bank of Australia; Intesa Sanpaolo", "investment_sector": "Energy, Power & Storage", "geography": "London, UK"}
                ]
            ),
            (
                """
                🛵 BluSmart Mobility, an Ahmedabad, India-based electric ride hailing company, raised $37M in Series A funding and $5M in debt financing from BP Ventures and Survam Partners.
                💸 Green-Got, a Paris, France-based green banking service, raised $5.5M in Seed funding from Pale Blue Dot and Equity crowdfunding through Crowdcube.
                """,
                [
                    {"name": "BluSmart Mobility", "stage": "Series A", "amount_raised": 37000000, "current_investors": "BP Ventures; Survam Partners", "investment_sector": "Transportation & Mobility", "geography": "Ahmedabad, India"},
                    {"name": "Green-Got", "stage": "Seed", "amount_raised": 5500000, "current_investors": "Pale Blue Dot; Equity crowdfunding through Crowdcube", "investment_sector": "General Catalyst", "geography": "Paris, France"}
                ]
            ),
            (
                """
                💨 Athian, an Indianapolis, IN-based livestock carbon marketplace, raised an undisclosed amount in Seed funding from DSM Venturing and California Dairies. 
                ☀️ Solar Ladder, an Andheri, India-based rooftop solar installation management platform, raised $1.4M in Seed funding from Aditya Bandi, Axilor Ventures, Deepak Jain, DevCo Partners, Subin Mitra, Titan Capital, and Varun Alagh. 
                """,
                [
                    {"name": "Athian", "stage": "Seed", "amount_raised": None, "current_investors": "DSM Venturing; California Dairies", "investment_sector": "Food, Agriculture & Forestry", "geography": "Indianapolis, IN"},
                    {"name": "Solar Ladder", "stage": "Seed", "amount_raised": 1400000, "current_investors": "Aditya Bandi; Axilor Ventures; Deepak Jain; DevCo Partners; Subin Mitra; Titan Capital; Varun Alagh", "investment_sector": "Energy, Power & Storage", "geography": "Andheri, India"}
                ]
            )

        ]
    )
    deal_chain = create_extraction_chain(llm, deal_scehema)

    max_deals = 20
    average_deal_length = 220 #in characters
    text = ""
    output = []
    
    for deal in scraper_deals:
        if len(text) > max_deals*average_deal_length:
            output += deal_chain.predict_and_parse(text=(text))['data']['deal']
            text = ""
        text += deal + "\n"
    if text == "":
        return []
    return output + deal_chain.predict_and_parse(text=(text))['data']['deal']
    

#EXITS:
def extract_exits(llm,scraper_exits):
    if len(scraper_exits) == 0:
        return []
    exit_schema = Object(
        # This what will appear in your output. It's what the fields below will be nested under.
        # It should be the parent of the fields below. Usually it's singular (not plural)
        id="exit",
        
        # Natural language description about your object
        description="Information about an exit",
        
        # Fields you'd like to capture from a piece of text about your object.
        attributes=[
            Text(
                id="name",
                description="The company name."
            ),
            Text(
                id="stage",description="The type of exit. MUST be one of the following: IPO; Acquisition; SPAC; Other; Undisclosed"
            ),
            Number(
                id="amount_raised",
                description="The value of the exit. MUST be expressed as an INTEGER (NOT STRING) with all included zeros and NO COMMAS or decimals. None if the amount is not mentioned."
            ),
            Text(
                id="current_investors",
                description="The company or companies making the acquisition if mentioned. Multiple investors MUST be separated by semi-colons."
            ),
            Text(
                id="investment_sector",
                description="The sector of the following that best fits the company: Buildings; Energy, Power & Storage; Waste, Plastic & Recycling; Food, Agriculture & Forestry; Industrial Processes & Management; Transportation & Mobility; General Catalyst; Other; Other: CCUS."
            ),
            Text(
                id="geography",
                description="The company location."
            )
        ],
        
        # Examples help go a long way with telling the LLM what you need
        examples=[
            (
                """ 
                ⚒️ American Battery Materials, a Greenwich, CT-based minerals exploration and development company focused on direct lithium extraction, announced a SPAC merger with Seaport Global Acquisition II Corp.
                💧Xylem, a Washington, D.C.-based water technology developer, completed its $7.5B all-stock acquisition of Evoqua Water Technologies.
                """, 
                [
                    {'name': 'American Battery Materials', 'stage': 'SPAC', 'amount_raised': None, 'current_investors': 'Seaport Global Acquisition II Corp', 'investment_sector': 'Energy, Power & Storage', 'geography': 'Greenwich, CT'},
                    {'name' : 'Evoqua Water Technologies', 'stage': 'Acquisition', 'amount_raised': 7500000000, 'current_investors': 'Xylem', 'investment_sector': 'Waste, Plastic & Recycling', 'geography': 'Washington, D.C.'}
                ]
            ),
            (
                """
                IrriWatch, a Netherlands-based provider of irrigation intelligence management software, was acquired by Hydrosat, a thermal data and satellite analytics company.
                ⚡ Brookfield Asset Management has acquired a controlling stake in CleanMax, a Mumbai, India-based renewable energy developer and operator, for $360M.
                """,
                [
                    {'name': 'IrriWatch', 'stage': 'Acquisition', 'amount_raised': None, 'current_investors': 'Hydrosat', 'investment_sector': 'Food, Agriculture & Forestry', 'geography': 'Netherlands'},
                    {'name': 'CleanMax', 'stage': 'Acquisition', 'amount_raised': 360000000, 'current_investors': 'Brookfield Asset Management', 'investment_sector': 'Energy, Power & Storage', 'geography': 'Mumbai, India'}
                ]
            ),
            (
                """
                ⚡ True Green Capital Management acquired a majority stake in CleanChoice Energy, a Washington, DC-based renewable energy supplier.
                ☀️ Evolar, a Uppsala, Sweden-based Perovskite tandem solar cell technology company, was acquired by First Solar for $38M (with additional payment up to $42M based on milestones).
                """,
                [
                    {'name': 'CleanChoice Energy', 'stage': 'Acquisition', 'amount_raised': None, 'current_investors': 'True Green Capital Management', 'investment_sector': 'Energy, Power & Storage', 'geography': 'Washington, DC'},
                    {'name': 'Evolar', 'stage': 'Acquisition', 'amount_raised': 38000000, 'current_investors': 'First Solar', 'investment_sector': 'Energy, Power & Storage', 'geography': 'Uppsala, Sweden'}
                ]
            )
        ]
    )
    exit_chain = create_extraction_chain(llm, exit_schema)

    #get exits from scraping
    text = ""
    for exit_announcement in scraper_exits:
        text += exit_announcement + "\n"
    #run extraction chain on exits
    return exit_chain.predict_and_parse(text=(text))["data"]['exit']

#clean up
def get_website(name):
    url = f"https://autocomplete.clearbit.com/v1/companies/suggest?query={name}"
    response = requests.get(url).json()
    if len(response) > 0:
        if 'domain' in response[0]:
            return response[0]['domain']
    else: return None

def ctvc_to_df(extracted_output,date,links):
    df = pd.DataFrame(extracted_output)
    df['date'] = date
    df['website'] = df['name'].apply(get_website)
    if len(links) == len(df):
        df['Funding Announcment'] = links 
    else: 
        df['Funding Announcment'] = None
        print("Number of links does not match number of deals")
    df = df.set_index('name')
    return df


