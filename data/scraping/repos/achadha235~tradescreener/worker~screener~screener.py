from dotenv import load_dotenv

load_dotenv()

import os
import requests
import urllib.parse

from colorama import Fore
from typing import List, Optional
from pandas import DataFrame
from pydantic import BaseModel

from llama_index.llms import OpenAI as LlamaOpenAI
from llama_index import ServiceContext

from langchain import PromptTemplate
from langchain.llms import OpenAIChat
from langchain.output_parsers import PydanticOutputParser

from books.book_index import load

api_key = os.getenv("INTRINIO_API_KEY")


class ScreenerResult(BaseModel):
    condition: str
    explanation: List[str]


class Screener(BaseModel, arbitrary_types_allowed=True):
    name: Optional[str]
    screener: str
    filter_conditions: str
    explanation: List[str]
    tickers: List[str]
    df: DataFrame


def fetch_for_condition(condition: str):
    results = []
    url = "https://api-v2.intrinio.com/securities/screen"
    current_page = 0
    params = {
        "primary_only": "true",
        "actively_traded": "true",
        "conditions": condition,
        "api_key": api_key,
        "page_number": current_page,
        "page_size": 1000,
    }
    query_url = url + "?" + urllib.parse.urlencode(params)
    response = requests.post(query_url, data=condition)

    all_data = response.json()

    for row in all_data:
        security = row["security"]
        data = row["data"]
        row = {
            "ticker": security["ticker"],
            "name": security["name"],
            "figi": security["figi"],
        }
        for d in data:
            val = d.get("number_value") or d.get("text_value")
            if d["tag"] != "country" and val:
                row[d["tag"]] = val
        results.append(row)
    return results


## Load up value investing book
book_index = load()

llama_llm = LlamaOpenAI(temperature=0, model="gpt-4")

gpt4 = OpenAIChat(temperature=0, model="gpt-4")
gpt3 = OpenAIChat(temperature=0, model="gpt-3.5-turbo")

service_context = ServiceContext.from_defaults(llm=llama_llm)

query_engine = book_index.as_query_engine(service_context=service_context)

screener_filters = """Stock Exchange, Industry Category, Industry Group, Sector, Close Price, Country, Revenue Q/Q Growth, Revenue Growth, Return on Common Equity, R&D to Revenue, Return on Invested Capital, Profit (Net Income) Margin, Return on Assets, Return on Equity, Price to Revenue, Quick Ratio, Price to Tangible Book Value, Operating Margin, Price to Earnings, Pre Tax Income Margin, Operating Expenses to Revenue, Price to Book Value, Operating Cash Flow to CapEx, Normalized NOPAT Margin, NOPAT Margin, Net Non-Operating Expense Percent, Long-Term Debt to Total Capital, Net Income Growth, Net Debt to EBITDA, Long-Term Debt to Equity, Market Capitalization, Long-Term Debt to EBITDA, Leverage Ratio, Invested Capital Growth, Gross Margin, EPS Growth, Financial Leverage, Enterprise Value to Operating Cash Flow, Enterprise Value to Revenue, Enterprise Value to EBITDA, Effective Tax Rate, EBIT to Interest Expense, Enterprise Value to EBIT, Enterprise Value, EBITDA Margin, EBIT Margin, Earnings Yield, Beta, Dividend Payout Ratio, Dividend Yield, Debt to Total Capital, Debt to EBITDA, Debt to Equity, Current Ratio, Altman Z-Score"""

screener_filter_tags = """
stock_exchange:String, industry_category:String, industry_group:String, sector:String, close_price:Float, country:String, revenueqoqgrowth:Percentage, revenuegrowth:Percentage,  roce:Percentage, rdextorevenue:Percentage, roic:Percentage, profitmargin:Percentage, roa:Percentage, roe:Percentage, pricetorevenue:Multiple, quickratio:Float, pricetotangiblebook:Multiple, operatingmargin:Percentage, pricetoearnings:Multiple, pretaxincomemargin:Percentage, opextorevenue:Percentage, pricetobook:Multiple, ocftocapex:Percentage, normalizednopatmargin:Percentage, nopatmargin:Percentage, nnep:Percentage, ltdebttocap:Percentage, netincomegrowth:Percentage, netdebttoebitda:Float, ltdebttoequity:Float, marketcap:USD, ltdebttoebitda:Float, leverageratio:Float, investedcapitalgrowth:Percentage, grossmargin:Percentage, epsgrowth:Percentage, finleverage:Float, evtoocf:Multiple, evtorevenue:Multiple, evtoebitda:Multiple, efftaxrate:Percentage, ebittointerestex:Float, evtoebit:Multiple, enterprisevalue:USD, ebitdamargin:Percentage, ebitmargin:Percentage, earningsyield:Percentage, beta:Float, divpayoutratio:Percentage, dividendyield:Percentage, debttototalcapital:Percentage, debttoebitda:Float, debttoequity:Float, currentratio:Float, altmanzscore:Float
"""

parser = PydanticOutputParser(pydantic_object=ScreenerResult)

design_screener_template = """\
You are a helpful financial advisor. Your client has access to a stock screener that can help them select stocks based on the following filters. 

{screener_filters}

The client as said this to you:
"{client_request}"

Based on the avaliable filters above and the client request, design a set of conditions using the filter above to select stocks that you would invest in. Make sure to explain your reasoning for each filter you select. Rely on your past value investing knowledge to determine appropriate values for each filter. If the user has mentioned any explicit preferences, make sure to incorporate them into your screener. Your screener should not be unrealistically strict, and should be able to return a reasonable number of stocks. 

Your final condition should be a single line of pseudocode representing a filtering condition that can be used to filter stocks in the screener. Example of a valid output: (Price to Revenue >= 1 and Price to Revenue <= 2) or (Market Capitalization >= 100,000,000,000 and (Altman Z-Score > 3 and Quick Ratio >= 1))
You must always use the condition (Country = United States of America) in your screener to filter for US-only stocks so that you never provide the user a non-US stock. You must NEVER return placeholders like "Industry Average" or "Industry Maximum" as filter placeholders values and must always use real numbers that are reasonable.
"""

name_prompt = """
Pick a short descriptive name for this screener. This name will be used to refer to this screener in the future.
You should base the name on the client's request and the filter conditions

Client Request:
{client_request}

Filter Conditions:
{screener_filters}

Return just the name of the screener without any additional text or quotation marks. Do not mention the country filter in the name.

Example: Undervalued, distressed mid-cap sttocks
"""


async def screen_stocks(client_request: str):
    """Screen stocks based on a client request."""

    ## Step 1: design a screener
    design_screener_prompt_template = PromptTemplate(
        template=design_screener_template,
        input_variables=["screener_filters", "client_request"],
    )
    print(Fore.BLUE + "Thinking of a screener that matches your conditions..." + "\n\n")

    output_natural_language = query_engine.query(
        design_screener_prompt_template.format(
            screener_filters=screener_filters, client_request=client_request
        )
    )

    ## Step 2: Take the natural language output and parse it into a JSON format
    parse_json_prompt = """
    Parse the following natural language text into the appropriate JSON format:

    {output_natural_language}

    {format_instructions}
    """

    parse_prompt_template = PromptTemplate(
        template=parse_json_prompt,
        input_variables=["output_natural_language"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    output = gpt4(
        parse_prompt_template.format(output_natural_language=output_natural_language)
    )

    # output = gpt4.complete(design_screener_prompt_template.format(screener_filters=screener_filters, client_request=client_request))
    result = parser.parse(output)
    condition = result.condition
    explanation = result.explanation

    print(Fore.BLUE + "You asked for: " + client_request + "\n\n")

    print(
        Fore.BLUE + "A good screener for this condition would be " + condition + "\n\n"
    )

    print(
        Fore.BLUE
        + "Explanation "
        + "\n".join(map(lambda e: "- " + e, explanation))
        + "\n\n"
    )

    ## Step 3: Name the screener

    name_prompt_template = PromptTemplate(
        template=name_prompt,
        input_variables=["client_request", "screener_filters"],
    )

    name = gpt3(
        name_prompt_template.format(
            screener_filters=condition, client_request=client_request
        )
    )

    ## Step 4: Construct and run the Intrinio API call
    screener_docs = ""
    with open(os.path.join(os.path.dirname(__file__), "screener-docs.md")) as f:
        screener_docs = f.read()

    example_screener = """{  "operator": "AND",  "clauses": [    {      "field": "country",      "operator": "eq",      "value": "United States of America"    }  ],  "groups": [    {      "operator": "NOT",      "clauses": [        {          "field": "pricetoearnings",          "operator": "gt",          "value": "5"        },        {          "field": "marketcap",          "operator": "gt",          "value": "2000000000"        }      ]    }  ]}"""
    generate_condition_prompt = """
    You are given the following requirements for a stock screener:
    {screener_condition}

    You must use the following documentation to construct a condition for a screener:
    {screener_docs}

    Here are the avaliable datatags you can filter with, along with their datatypes, represented as a comma-separated list of [datatag]:[unit]

    {screener_filter_tags}

    IMPORTANT RULES:
    - You must output only the JSON body required for the filter
    - You must always add a final filter to only return US stocks
    - You must only use the avaliable datatags
    - Any percentage values must be expressed as decimals - e.g. 10% must be expressed as 0.1
    - Do not add any words to explain your reasoning or present the output. Only return filter JSON as a single line of text without any additional or unnecessary text, explanations, formatting, punctuation, whitespace etc.

    Example of a valid output:
    {example_screener_condition}


    """

    generate_condition_prompt_template = PromptTemplate(
        template=generate_condition_prompt,
        input_variables=["screener_condition"],
        partial_variables={
            "screener_filter_tags": screener_filter_tags,
            "screener_docs": screener_docs,
            "example_screener_condition": example_screener,
        },
    )

    filter_conditions = gpt4(
        generate_condition_prompt_template.format(screener_condition=condition)
    )

    print(Fore.BLUE + "Filter Condition: " + filter_conditions + "\n\n")

    stocks = fetch_for_condition(filter_conditions)
    df = DataFrame.from_dict(stocks)

    print(Fore.BLUE + "Screened stocks: " + df.to_markdown())

    return Screener(
        name=name,
        df=df,
        tickers=list(df["ticker"]),
        explanation=explanation,
        screener=condition,
        filter_conditions=filter_conditions,
    )
