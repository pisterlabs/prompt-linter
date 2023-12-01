"""
A collection of functions used to extract data from a PDF file and return a JSON object for the given schema under term-sheets.
"""

from rag import turn_path_to_json
from pydantic import BaseModel, Field
from typing import List, Optional
import json
from langchain.document_loaders import PyPDFLoader
from openai import OpenAI
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import pandas as pd

class Beta(BaseModel):
    Isin: str = Field(..., description="Unique identifier for the structured product, following the International Securities Identification Number (ISIN) format.")
    Issuer: str = Field(..., description="Name of the entity issuing the structured product. This should be the full legal name of the issuer.")
    Ccy: str = Field(..., description="The three-letter currency code representing the currency of the product, as per ISO 4217 standard. Example: 'EUR'.")
    Underlying: List[str] = Field(..., description="List of underlying assets or indices associated with the product. Provide up to five valid tickers. Example: ['SX5E', 'UKX', 'SPX'].")
    Launchdate: str = Field(..., description="The launch or initial valuation date of the product, marking the start of its lifecycle, in 'dd/mm/yyyy' format. Example: '31/12/2021'. This date sets the initial conditions for the product. Also called the Trade Date or Initial Valuation Date. ")
    Maturity: str = Field(..., description="The maturity date of the product, indicating its expiration and the end of its term, in 'dd/mm/yyyy' format. It's the date when final settlements are made based on the final valuation. Example: '31/12/2023'.")
    Barrier: int = Field(..., description="The barrier level of the product, specified in percentage terms. This represents a critical price level for features like knock-in. Example: 70 (indicating 70% of the initial price).")



def betas_to_csv(items: list, file_name : str) -> None:
    """
    Takes a list of Beta objects and saves them to a csv file with the given name.
    """
    beta_field_to_csv = {
        "Isin": "Isin",
        "Issuer": "Issuer",
        "Ccy": "Ccy",
        "Underlying": "Underlying(s)",
        "Strike": "Strike",
        "Launchdate": "Launch Date",
        "Finalvalday": "Final Val. Day",
        "Maturity": "Maturity",
        "Barrier": "Barrier",
        "Cap": "Cap"
    }
    # some items might be missing fields
    # we need to add them
    for item in items:
        for field in beta_field_to_csv:
            if field not in item:
                item[field] = "Nan"

    # maintain the order of the fields as specified in the schema
    beta_field_order = beta_field_to_csv.keys()
    # create a dataframe
    df = pd.DataFrame(items, columns=beta_field_order)
    # rename to dict vals
    df.rename(columns=beta_field_to_csv, inplace=True)
    # save it to a csv file
    df.to_csv(file_name, index=False)








# keywords = ['isin','issuer','ccy','currency','underlying','underlyings','strike','strikes','launch','date','dates','final valuation','day','maturity','cap','barrier','redemption','amount']
keywords = [
    'issuer', 'issuing','issuing entity', 'issuing company', 'issuing corporation', 'issuer firm', 'issuing institution',
    'currency', 'ccy', 'money','monetary', 'monetary unit', 'legal tender', 'fiat currency', 'exchange medium',
    'underlying', 'assests' 'underlying assets', 'base assets', 'core assets', 'fundamental assets',
    'strike date', 'strike day', 'exercise date', 'option strike date', 'option exercise date', 'strike',
    'final valuation date', 'last valuation date', 'ultimate valuation date', 'end valuation date',
    'launch date', 'start date', 'inception date', 'commencement date', 'beginning date', 'opening date',
    'maturity date', 'expiration date', 'expiry date', 'termination date', 'end date', 'last date', 'due date',
    'isin', 'international securities identification number', 'security identifier', 'stock identifier','instrument identifier',
    'strike', 'strikes', 'strike price', 'exercise price', 'option price', 'target price',
    'laung','launch date', 'initiation date', 'start date','inception date' 'commence launch', 'begin launch', 'inaugurate launch',
    'date', 'dates', 'day', 'days','time', 'period', 'periods', 'moment', 'calendar day',
    'final valuation', 'last valuation', 'ultimate valuation', 'final assessment', 'end valuation',
    'business day', 'trading day', 'working day',
    'cap','cap level','boundary', 'ceiling', 'limit', 'maximum', 'upper bound', 'upper limit','top level',
    'barrier', 'threshold', 'limit', 'boundary', 'obstacle', 'hindrance', 'trigger level','barrier point',
    # hard coded values
    'percent', 'max', ' x ', ' × ', 'redemption date', 'redemption amount', 'usd', 'eur', 'barrier event',
    "%"
]



def count_words(text : str) -> int:
    """
    Counts the number of words in a string.
    """
    words = re.findall(r'\w+', text)
    return len(words)


def count_file_words(data : list) -> int:
    """
    Counts the number of words in a list of pages.
    """
    word_count = 0
    for page in data:
        word_count += count_words(page.page_content)
    print(word_count)
    return word_count


def format_response_to_json(response_string : str, gpt4 : bool = False) -> dict:
    """
    Takes a string and formats it into a JSON object. This is used to parse the output of the previous model.
    """
    client = OpenAI()
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106" if not gpt4 else "gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": "You are an assistant specialized in financial data analysis and extraction. Your task is to meticulously process a structured product schema and accurately populate a form with relevant data extracted from a provided document. It is your job to to extract a solid JSON from the provided message. If any values are speculative or uncertain, you should not include them in the JSON. If anything is yet to be extracted, ignore it."
            },
            {
                "role": "user",
                "content": "This is the message you need to extract a JSON from: " + response_string
            },
            {
                "role": "user",
                "content": "The following are fields that need to be extracted from the document: " + Beta.schema_json(indent=2)
            },
            {
                "role": "user",
                "content": "Think carefully and think step by step. Take a deep breath and give me an accurate JSON. DO NOT create any new fields. If you are not sure about a value, leave it blank."
            }
        ],
        response_format={'type': "json_object"}
    )
    data = completion.choices[0].message.content
    parsed = json.loads(data)
    return parsed


def extract_data(file_name : str, gpt4: bool = False) -> dict:
    """
    Extracts data from a PDF file and returns a JSON object.
    """
    questions = [
        "Can you list the strike prices for each underlying asset for this product? The strike price is the set price at which an option contract can be bought or sold when it is exercised.",
        "What is the final valuation day for this product? This is the date set for the final assessment of the product's value before its maturity.",
        "Is there a cap on the product's return mentioned in the document? If so, what is it? The cap is the maximum limit on the return that the product can generate.",
    ]
    # strike, final valuation, cap

    # ['completed', 'completed', 'completed', 'completed', 'in_progress', 'completed', 'in_progress', 'completed', 'in_progress', 'completed']


    hard = turn_path_to_json(file_name, questions)
    client = OpenAI()
    path =  file_name
    loader = PyPDFLoader(path)
    data=loader.load()
    #r'\b(?:\d{1,2}[-\/.]\d{1,2}[-\/.]\d{2,4}|\d{2,4}[-\/.]\d{1,2}[-\/.]\d{1,2}|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}[,.]?[-\s]*\d{2,4}|\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,.\s]+\d{2,4})\b'

    stop_words = set(stopwords.words('english'))
    regex_pattern = r'\b(?: '+'|'.join(map(re.escape, keywords)) + r')\b|\b(?:\d{1,2}[-\/.]\d{1,2}[-\/.]\d{2,4}|\d{2,4}[-\/.]\d{1,2}[-\/.]\d{1,2}|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}[,.]?[-\s]*\d{2,4}|\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,.\s]+\d{2,4})\b'
    seen = set()
    raw = ""

    # TODO Here hte issue is that we are minifying all the pages, which is not optimal
    # we should check if a whole document is too long, and only then minify it
    # We might be able to do this quickly with the document object but im not sure
    if gpt4 or count_file_words(data) < 10000:
        # pass everything to the model
        for page in data:
            hasOccurence = page.page_content is not None
            shouldAdd = hasOccurence is not None
            if shouldAdd:
                raw += page.page_content + " "
    else:
        print("Minifying")
        # trim the data
        for page in data:
            filtered_page = re.search(regex_pattern, page.page_content, re.IGNORECASE)
            hasOccurence = filtered_page is not None
            shouldAdd = hasOccurence is not None
            if shouldAdd:
                raw += page.page_content + " "


        raw = raw.replace("\n", " ")

        # add stop words
        tokenized_raw = word_tokenize(raw)
        raw = ""
        for w in tokenized_raw:
            if w not in stop_words:
                raw += w

    print("New length: ", count_words(raw))

    print("Running Query")

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106" if not gpt4 else "gpt-4-1106-preview",
        messages=[
            {
                "role": "system",
                "content": "You are an assistant specialized in financial data analysis and extraction. Your task is to meticulously process a structured product schema and accurately populate a form with relevant data extracted from a provided document."
            },
            {
                "role": "user",
                "content": "The structured product schema is defined as follows:" + Beta.schema_json(indent=2)
            },
            {
                "role": "user",
                "content": "Here is the document with the necessary data:"
            },
            {
                "role": "user",
                "content": raw
            },
            {
                "role": "user",
                "content": "Please extract the data from the document"
            }
        ],
    )
    # get the status of the completion
    print(completion)
    # combine the data
    combined = {}

    ct = completion.choices[0].message.content
    parsed = format_response_to_json(ct, gpt4=gpt4)

    for key in parsed:
        combined[key] = parsed[key]

    for key in hard:
        combined[key] = hard[key]

    return combined
