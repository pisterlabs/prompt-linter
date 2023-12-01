import os
from dotenv import load_dotenv
import requests
import openai
import ast

# Load environment variables
load_dotenv()

EBAY_API_KEY = os.getenv('EBAY_API_KEY')

functions = [
    {
        'name': 'search_product',
        'description': 'Search for a product.',
        'parameters': {
            'type': 'object',
            'properties': {
                'keywords': {
                    'type': 'string',
                    'description': 'Keywords to search for a product.'
                },
                'min_price': {
                    'type': 'number',
                    'description': 'Minimum price for the product search.'
                },
                'max_price': {
                    'type': 'number',
                    'description': 'Maximum price for the product search.'
                },
                'free_shipping': {
                    'type': 'string',
                    'description': 'Whether the product should have free shipping (true or false).'
                }
            }
        },
        'required': ['keywords']
    }
]


def get_products_from_prompt(prompt):
    messages = [
        {'role': 'system', 'content': 'You are a helpful shopping assistant. You will extract the key information from the user\'s message and use the functions provided to you to finish the task.'},
        {'role': 'user', 'content': prompt},
    ]

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo-0613',
        messages=messages,
        functions=functions,
        function_call='auto'
    )

    arguments = ast.literal_eval(
        response['choices'][0]['message']['function_call']['arguments'])

    headers = {
        "X-EBAY-SOA-SECURITY-APPNAME": EBAY_API_KEY,
        "X-EBAY-SOA-OPERATION-NAME": "findItemsByKeywords",
        "X-EBAY-SOA-SERVICE-VERSION": "1.0.0",
        "X-EBAY-SOA-RESPONSE-DATA-FORMAT": "JSON",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    url = 'https://svcs.ebay.com/services/search/FindingService/v1'
    params = {
        "keywords": arguments['keywords']
    }

    # Initialize the itemFilter index
    item_filter_index = 0

    # Check if min_price is provided and add it to the params
    if 'min_price' in arguments:
        min_price_filter = f"itemFilter({item_filter_index})"
        params[f"{min_price_filter}.name"] = "MinPrice"
        params[f"{min_price_filter}.value"] = str(arguments['min_price'])
        params[f"{min_price_filter}.paramName"] = "Currency"
        params[f"{min_price_filter}.paramValue"] = "USD"
        item_filter_index += 1

    # Check if max_price is provided and add it to the params
    if 'max_price' in arguments:
        max_price_filter = f"itemFilter({item_filter_index})"
        params[f"{max_price_filter}.name"] = "MaxPrice"
        params[f"{max_price_filter}.value"] = str(arguments['max_price'])
        params[f"{max_price_filter}.paramName"] = "Currency"
        params[f"{max_price_filter}.paramValue"] = "USD"
        item_filter_index += 1

    # Check if free_shipping is provided and add it to the params
    if 'free_shipping' in arguments:
        free_shipping_filter = f"itemFilter({item_filter_index})"
        params[f"{free_shipping_filter}.name"] = "FreeShippingOnly"
        params[f"{free_shipping_filter}.value"] = arguments['free_shipping']

    return requests.get(url, headers=headers, params=params).json()
