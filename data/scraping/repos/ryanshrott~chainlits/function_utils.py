import openai
import json, ast
import os
import chainlit as cl
from dotenv import load_dotenv
import openai
import os
from funkagent import agents, parser
import inspect
import re
import psycopg2
load_dotenv(".env")
openai.api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
from helper_utils import *
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
import requests
import googlemaps
from chainlit import AskUserMessage, Message, on_chat_start
from function_utils import *
from langchain.chains import TransformChain, LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate, ChatPromptTemplate
openai.api_key = os.environ.get("OPENAI_API_KEY")
from langchain.chains import create_tagging_chain, create_tagging_chain_pydantic
from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Optional
gmaps = googlemaps.Client(key=os.getenv("GOOGLE_KEY"))
from news import news
llm_random = ChatOpenAI(temperature=0.5)
class PersonalDetails(BaseModel):
    name: Optional[str] = Field(
        None,
        description="The name of the client. Include surname if possible, i.e. John Smith."
    )
    city: Optional[str] = Field(
        None,
        description="The name of the city where the client is looking to make a real estate transaction. Ensure it's a real city in Canada and spelled correctly. i.e. Toronto"
    )
    preferred_language: Optional[str] = Field(
        None,
        description="The language that the person prefers to communicate in, i.e. English, French, Spanish, etc."
    )
    email: Optional[str] = Field(
        None,
        description="The email address that client associates as theirs i.e. johndoe@gmail.com"
    )
    phone: Optional[str] = Field(
        None,
        description="The phone number that the client associates as theirs (i.e. 647-972-7058)"
    )




def str_to_bool(s):
    if s.lower() == 'true':
         return True
    elif s.lower() == 'false':
         return False
    else:
         raise ValueError("Cannot convert to boolean: " + s)

def geocoords(place_id):
    place_details = gmaps.place(place_id)
    latitude = place_details['result']['geometry']['location']['lat']
    longitude = place_details['result']['geometry']['location']['lng']
    return {'lat': latitude, 'lon': longitude}


def get_gmaps_address(query):
    return gmaps.places_autocomplete(query, types=["geocode"], components={"country": "ca"})


def get_address_info(top_suggestion, unit_number=None):
    coords = geocoords(gmaps.find_place(top_suggestion, 'textquery')['candidates'][0]['place_id'])
    return {
        'description': top_suggestion['description'],
        'street_num': top_suggestion['terms'][0]['value'],
        'street_name': top_suggestion['terms'][1]['value'].split(' ')[0],
        'city': top_suggestion['terms'][2]['value'],
        'unit_number': unit_number,
        'lat': coords['lat'],
        'lon': coords['lon']
    }

def get_googlemaps_address_geocoords_info(address: str, unit_number: str='') -> str:
    """Get the address, geocoordinates, street number, street name, city and unit number for a given address. 

    :param address: The address of the property (e.g. 123 Harvie Ave Toronto, Ontario)
    :param unit_number: The unit number like 402 (optional)

    :return: The address, geocoordinates, street number, street name, city and unit number for a given address. 
    """
    suggestion = get_gmaps_address(address)
    info = get_address_info(suggestion[0], unit_number)
    return str(info)

def price_home_and_find_comparables(address: str, house_category: str='Detached',  unit_number:str='', bedrooms:str='', washrooms:str='', house_area_sqft:str='', land_area:str='', use_comparables:str="false"):
    """
    Call the SmartBids.ai pricing model used to compute the model price of a home. You can also optionally find comparable properties and price them as well.    

    :param address: The address of the property (e.g. 123 Harvie Ave, Toronto, ON)
    :param house_category: The type/category of the property like Detached or Condo etc.. Default is detached if not specified. <<Condo, Detached, Semi, Link, Multiplex, Vacant Land, Condo Townhouse, Freehold Townhouse, Other>>
    :param unit_number: The unit number like 402 (optional)
    :param bedrooms: The number of bedrooms (optional)
    :param washrooms: The number of bathrooms (optional)
    :param house_area_sqft: The square footage of the house (optional)
    :param land_area: The square footage of the land (optional)
    :param use_comparables: Whether to use comparables or not (optional). Do NOT use unless user asks for comparables. If set to true, you MUST specify at least one of bedrooms, washrooms, house_area_sqft, land_area. <<true, false>>

    :return: Pricing information for the property
    """
    print('address', address)
    print('house_category', house_category)
    print('unit_number', unit_number)
    print('bedrooms', bedrooms)
    print('washrooms', washrooms)
    print('house_area_sqft', house_area_sqft)
    print('land_area', land_area)
    print('use_comparables', str_to_bool(use_comparables))
    headers = {
        'Authorization': os.getenv("HUGGING_FACE_API_KEY"),
        'Content-Type': "application/json"
    }
    if(str_to_bool(use_comparables) and (bedrooms == '' and washrooms == '' and house_area_sqft == '' and land_area == '')):
        return 'You must specify at least one of bedrooms, washrooms, house_area_sqft, land_sqft if you want to use comparables'
    data = [
            address,
            house_category,
            unit_number,
            str_to_bool(use_comparables),
            True,
            bedrooms,
            washrooms,
            house_area_sqft,
            land_area,
            '',
            {'headers': ['None'], 'data': [['None']]}
        ]
    print(data)
    #print({'headers': ['None'], 'data': [['None']]} if modifications is None else modifications)
    response = requests.post("https://rshrott-smartbids.hf.space/run/price_with_google_maps", json={
        "data": data
    }, headers=headers)


    if response.status_code == 200:
        data = response.json()["data"]
        output_str = parse_json_model_output(data)
        print(output_str)
        return output_str
    else:
        return 'pricing error'
    
def parse_json_model_output(json_data):
    total_weight = 0
    weighted_sum = 0
    house_data = []
    comparable_properties = []

    exact_matches = 0
    for data in json_data:
        for key, value in data.items():
            if key == "input_info": # Ignore "input_info" data
                input_info = value
                continue

            weight = value['weight']
            first_price = value['prices'][next(iter(value['prices']))]['price_simple']
            beds = value['total_beds']
            washrooms = value['house-washroom']
            area_estimate = value['house-house_area-estimate']
            land_area = value['house-land-area']
            exact_match = value['exact_match']

            house_data.append({
                "street": key,
                "price": first_price,
                "beds": beds,
                "washrooms": washrooms,
                "area_estimate": area_estimate,
                "land_area": land_area,
                "exact_match": exact_match,
            })

            total_weight += weight
            weighted_sum += weight * first_price

            if exact_match:
                exact_matches += 1

    weighted_avg = weighted_sum / total_weight if total_weight > 0 else 0
    location_coords = (input_info['lat'], input_info['lon'])

    streets = ', '.join([house['street'] for house in house_data])
    for house in house_data:
        comparable_properties.append(f"{house['street']},{house['price']:.2f},{house['beds']},{house['washrooms']},{house['area_estimate']},{house['land_area']}")
    comparable_properties_csv = "\n".join(comparable_properties)

    if len(house_data) == 1 and exact_matches == 1:  # Only one exact match found
        house = house_data[0]
        result_str = f"The model price of {input_info['street_num']} {input_info['street_name']} is {house['price']:.2f}. The location coordinates are {location_coords}. The house has {house['beds']} beds, {house['washrooms']} washrooms, an estimated area of {house['area_estimate']}, and a land area of {house['land_area']}. We priced this with an exact match."
    elif exact_matches == len(house_data): # All houses are exact matches
        result_str = f"The model price of {input_info['street_num']} {input_info['street_name']} is {weighted_avg:.2f}. The location coordinates are {location_coords}. The house has {house['beds']} beds, {house['washrooms']} washrooms, an estimated area of {house['area_estimate']}, and a land area of {house['land_area']}. We priced this with an exact match. You could also price this property with comparables: {streets}.\nThe comparable properties used are (in CSV format):\nStreet,Price,Beds,Washrooms,Area Estimate,Land Area\n{comparable_properties_csv}"
    else: # Not all houses are exact matches, we have comparables
        result_str = f"The model price of {input_info['street_num']} {input_info['street_name']} is {weighted_avg:.2f}. The location coordinates are {location_coords}.This price was computed using comparable properties: {streets}. \nThe comparable properties used are (in CSV format):\nStreet,Price,Beds,Washrooms,Area Estimate,Land Area\n{comparable_properties_csv}"

    return result_str


def calculator(num1: float, num2: float, operator: str) -> float:
    """Basic calculator function for +, -, *, /.

    :param num1: first number
    :param num2: second number
    :param operator: operator <<+, -, *, />>
    :return: result of the calculation
    """
    allowed_operators = {'+', '-', '*', '/'}

    if operator not in allowed_operators:
        raise ValueError(f'Invalid operator. Allowed operators are {allowed_operators}')

    if operator == '+':
        result = num1 + num2
    elif operator == '-':
        result = num1 - num2
    elif operator == '*':
        result = num1 * num2
    elif operator == '/':
        if num2 == 0:
            raise ZeroDivisionError('Division by zero is not allowed.')
        result = num1 / num2
    return str(result)

persist_directory = 'db'
# Now we can load the persisted database from disk, and use it as normal. 
vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=OpenAIEmbeddings())
retriever = vectordb.as_retriever(search_kwargs={"k": 10})

# create the chain to answer questions 
# Set up the turbo LLM
turbo_llm = ChatOpenAI(
    temperature=0,
    model_name='gpt-3.5-turbo-0613'
)
'''
# create the chain to answer questions 
#qa_chain = RetrievalQA.from_chain_type(llm=turbo_llm, 
#                                  chain_type="stuff", 
#                                  retriever=retriever, 
#                                  return_source_documents=True)

def get_realestate_news_vector_db(question : str) -> str:
    """Get the latest news about the real estate market in Canada. Only use when user asks a specific question about current news.
    :param question: ask about current real estate news
    :return: response and the sources of the information
    """
    llm_response = qa_chain(question)
    print(llm_response)
    sources = []
    for source in llm_response["source_documents"]:
        sources.append(source.metadata['source'])
    return f"Result:\n{llm_response['result']}.\nSources:\n{sources}"
'''
def get_realestate_news(question : str) -> str:
    """Get the latest news about the real estate market in Canada. Only use when user asks a specific question about current news.
    :param question: ask about current real estate news
    :return: response and the sources of the information
    """
    prompt = f'''Answer the following real estate market news question by using the news summary report below. If you can't answer based on the summary, say that you don't know and should contact a local realtor.
    Question: {question}
    Current News Summary:\n {news} \n
    Answer: 
    '''
    # Call open ai
    MODEL = "gpt-3.5-turbo"
    response = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    message = response['choices'][0]['message']['content']
    return message

conn_string = os.getenv("POSTGRES_URI_MODEL")
conn = psycopg2.connect(conn_string)
cursor = conn.cursor()

database_info_string = get_database_info(cursor)
cursor.close()
conn.close()
description=f"""SQL query used to extract info in order to answer user query. SQL query should be written using this database schema information:\n
{database_info_string}\nTips: 
Tip 1.The query should be returned in plain text, not in JSON. Do not use new lines characters inside the query. 
Tip 2. Always use the ILIKE operator when searching by a string in your query. For example use WHERE community_name ILIKE '%Annex%' instead of WHERE community_name='Annex'. 
Tip 3. When searching for addresses with unit numbers do like WHERE address ILIKE '%318 Richmond St W%' AND address ILIKE '%1105%'. 
Tip 4. Always limit yourself to 5 results. For example use LIMIT 5 at the end of your query. 
Tip 5. A good deal on a home is defined as a property with a high model to list price ratio.
Tip 6. NEVER run a SELECT * because it will return too much data. Only query columns that you need to answer the user's question. For example, if the user asks for the price of a property, you do not need to query the number of bedrooms, bathrooms, etc.
Tip 7: This table only contains current sales data, and NO historical data. If the user asks for historical data, refer them to app.smartbids.ai for more information.
Tip 8: To find nearby properties, you can use something like SELECT address, model_price FROM chat WHERE ST_DistanceSphere(ST_MakePoint(-79.4523143, 43.6817537), ST_MakePoint(lon, lat)) <= 500.
Tip 9: When searching for a community like "Trinity-Bellwoords" or "Corso Italia-Davenport", just search by a single word, i.e. WHERE community_name ILIKE '%Trinity%' or WHERE community_name ILIKE '%Corso%' or WHERE community_name ILIKE '%Annex%'
EXAMPLES of queries:
"SELECT address, listing_price, bedrooms, washrooms, house_area_sqft FROM chat WHERE community_name ILIKE '%Beaches%' AND house_category = 'Detached' AND bedrooms >= 4 AND house_area_sqft >= 2500 LIMIT 5"
"SELECT address, model_price, listing_price, bedrooms, house_area_sqft FROM chat WHERE community_name ILIKE '%Beaches%' AND house_category = 'Detached' AND bedrooms >= 4 AND house_area_sqft >= 2500 ORDER BY model_price/listing_price DESC LIMIT 5"
"""

docstring = f"""Query the database of Canadian realestate properties currenty for sale with a SQL query. This database has NO historical data, ONLY properties currently for sale.  
:param query: {description}
:return: result of the query"""
def query_realestate_database(query: str) -> str:
    try:
        conn_string = os.getenv("POSTGRES_URI_MODEL")
        conn = psycopg2.connect(conn_string)
        cursor = conn.cursor()
        cursor.execute(query)
        results = cursor.fetchall()
        cursor.close()
        conn.close()
        return str(results)
    except Exception as e:
        return f"Error: {e}"
query_realestate_database.__doc__ = docstring


def send_email_to_client(subject: str, message: str) -> str:
    """use when the client asks to meet in person or needs to book a showing. Also use when the conversation is over, or when you need help or are stuck.

    :param subject: The subject of the email
    :param message: A formal email message body addressed to the clients first name, which summarizes the conversation with the client and their current needs. Your formal signature should be included at the end.

    :return: A message indicating whether the email was sent successfully or not.
    """
    client_email = cl.user_session.get("person_details").email
    print('Trying to send email...')

    from_address = 'ryan@smartbids.ai'
    password = os.getenv("EMAIL_PASS")

    # Compose message
    msg = MIMEMultipart()
    msg['From'] = from_address
    msg['To'] = client_email + 'ryans664@gmail.com'
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    try:
        with smtplib.SMTP_SSL('mail.privateemail.com', 465) as server:
            server.login(from_address, password)
            text = msg.as_string()
            server.sendmail(from_address, client_email, text)
            return "Inform the client that an email has been sent."
    except Exception as e:
        return f'Failed to send email to {client_email} due to {str(e)}'
    
def calculate_monthly_mortgage_payment(principal: float, annual_interest_rate: float, loan_duration_in_years: float) -> float:
    """compute the monthly mortgage payment for a given principal, annual interest rate and loan duration in years.
    :param principal: The principal amount of the loan
    :param annual_interest_rate: The annual interest rate of the loan
    :param loan_duration_in_years: The duration of the loan in years
    :return: The monthly mortgage payment
    """
    if annual_interest_rate == 0:
        return principal / (loan_duration_in_years * 12)

    monthly_interest_rate = annual_interest_rate / (100 * 12)
    number_of_payments = loan_duration_in_years * 12

    # formula to calculate monthly payments
    monthly_payment = principal * (monthly_interest_rate * (1 + monthly_interest_rate)**number_of_payments) \
                       / ((1 + monthly_interest_rate)**number_of_payments - 1)

    return str(monthly_payment)

def calculate_land_transfer_tax(property_value: int, down_payment_amount: int=0, region_code: str="ON", first_time_buyer: str="false", new_construction: str="false", city: str = 'Toronto') -> dict:
    """Calculate the land transfer tax for a given property value, down payment amount, region code, first time buyer status, new construction status and city.
    :param property_value: The property value
    :param down_payment_amount: The down payment amount
    :param region_code: The region code <<ON, BC, AB>>
    :param first_time_buyer: Whether the buyer is a first time buyer or not <<true, false>>
    :param new_construction: Whether the property is a new construction or not <<true, false>>
    :param city: The city of the property

    :return: The land transfer tax
    """

    url = "https://widget.nesto.ca/api/v1/calculators/landtransfertax"
    print(property_value, down_payment_amount, region_code, first_time_buyer, new_construction, city)
    print(type(property_value), type(down_payment_amount), type(region_code), type(first_time_buyer), type(new_construction), type(city))
    payload = {
        "propertyValue": int(property_value),
        "downPaymentAmount": int(down_payment_amount),
        "regionCode": region_code,
        "firstTimeBuyer": str_to_bool(first_time_buyer),
        "newConstruction": str_to_bool(new_construction),
        "city": city
    }

    headers = {
        "authority": "widget.nesto.ca",
        "accept": "application/json",
        "accept-language": "en-US,en;q=0.9",
        "api-key": "6sLx0ZcZAXcpXa5OU8wp2c6IDbejRevtrCEiOdDFjeY5",
        "content-type": "application/json",
        "origin": "https://widget.nesto.ca",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }

    response = requests.request("PUT", url, json=payload, headers=headers)

    if response.status_code == 200:
        return str(response.json())
    else:
        return "Error: " + str(response.json())


functions = [
    {
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location"],
        },
    }
]

functions=[calculator, send_email_to_client, calculate_monthly_mortgage_payment, calculate_land_transfer_tax, price_home_and_find_comparables, get_realestate_news, query_realestate_database]
functions = [parser.func_to_json(func) for func in functions]
print(functions)

sys_msg = """Your are a friendly Canadian real estate agent named Ryan Shrott working for SmartBids.ai in September 2023. You are trying to help a client buy and/or sell a property. Before you begin chatting or answering questions, you MUST collect your client's first name and email address. Always address your client by their first name. When responding to a client, always follow up with another question in order to better hone in on their desires/needs. You have access to the following tools:
Tool 1: SQL database of curently listed properties. You do not have access to historical data. If a client asks for historical data, refer them to app.smartbids.ai for more information. 
Tool 2: The propietary SmartBids.ai pricing model, which you can use to price a home and optionally find comparable properties. ALWAYS assume that 'price_simple' in the JSON ouput is the model price of the property.
Tool 3: A function which finds comparable properties and pricing them. You can use this function to find and price comparable properties based on the street number, street name, city, house type, and optional parameters.
Tool 4: A news search database which you can ask targeted questions at to learn about the current real estate market news/trends. ONLY use this if client specifically asks for current news/trends.
Tool 5: A mortgage payments calculator.
Tool 6: A land transfer tax calculator.
Tool 7: The ability to send emails to clients. You MUST first aquire the client email. Use when the client asks to meet in person or needs to book a showing. Also use when the conversation is over, or when you recieve some errors.
Tool 8: A basic calculator that can perform addition, subtraction, multiplication and division.

Tips:
Tip 1. Before you send an email to a client, you must first aquire their email address.
Tip 2. Prioritize collecting the client's email address and first name before you answer their questions in detail.
Tip 3. Don't use the news tool unless the client specifically asks for current news/trends.
"""


def handle_function_request(function_name, arguments):
    if function_name == "get_realestate_news":
        question = arguments.get("question")
        if not question or question=='':
            return 'your input for question was empty'
        function_response = get_realestate_news(question=arguments.get("question"))
    elif function_name == "query_realestate_database":
        function_response = query_realestate_database(query=arguments.get("query"))
    elif function_name == "calculator":
        function_response = calculator(
            num1=arguments.get("num1"),
            num2=arguments.get("num2"),
            operator=arguments.get("operator"),
        )
    elif function_name == "send_email_to_client":
        function_response = send_email_to_client(
            subject=arguments.get("subject"),
            message=arguments.get("message"),
        )
    elif function_name == "calculate_monthly_mortgage_payment":
        function_response = calculate_monthly_mortgage_payment(
            principal=arguments.get("principal"),
            annual_interest_rate=arguments.get("annual_interest_rate", 5.25),
            loan_duration_in_years=arguments.get("loan_duration_in_years", 30),
        )
    elif function_name == "calculate_land_transfer_tax":
        function_response = calculate_land_transfer_tax(
            property_value=arguments.get("property_value"),
            down_payment_amount=arguments.get("down_payment_amount", 0.0),
            region_code=arguments.get("region_code", "ON"),
            first_time_buyer=arguments.get("first_time_buyer", "false"),
            new_construction=arguments.get("new_construction", "false"),
            city=arguments.get("city", 'Toronto'),
        )
    elif function_name == "price_home_and_find_comparables":
            print(arguments)
            print(arguments.get("house_category", 'Detached'))
            function_response = price_home_and_find_comparables(
                address=arguments.get("address"),
                house_category=arguments.get("house_category", 'Detached'),
                unit_number=arguments.get("unit_number", ""),
                bedrooms=arguments.get("bedrooms", ""),
                washrooms=arguments.get("washrooms", ""),
                house_area_sqft=arguments.get("house_area_sqft", ""),
                land_area=arguments.get("land_area", ""),
                use_comparables=arguments.get("use_comparables", "false"),
            )
    elif function_name == "get_googlemaps_address_geocoords_info":
        function_response = get_googlemaps_address_geocoords_info(
            address=arguments.get("address"),
            unit_number=arguments.get("unit_number", ""),
        )
    else:
        function_response = {"error": f"Unknown function: {function_name}"}

    return function_response