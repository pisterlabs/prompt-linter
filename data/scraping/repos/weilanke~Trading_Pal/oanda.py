import openai
import requests
import csv
import os
import ast

from words import trading_keywords, endpoint_phrases, messages, intents


# Set the OpenAI and OANDA API keys
OPENAI_API_KEY = "your api here"
OANDA_API_KEY = "your api here"
openai.api_key = OPENAI_API_KEY

# Set the base URL for the OANDA API
BASE_URL = "https://api-fxpractice.oanda.com"
ACCOUNT_ID = "your id here"

# The headers for the HTTP requests
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {OANDA_API_KEY}",
    "Connection": "keep-alive"
}
# Maximum token limit for each conversation
MAX_TOKENS = 4096

# Function to check if user input is trading-related
def is_trading_related(user_input):
    # Convert the user's input to lowercase
    user_input = user_input.lower()

    # Check if any of the trading keywords are in the user's input
    for keyword in trading_keywords:
        if keyword in user_input:
            return True

    # If no trading keywords were found in the user's input, return False
    return False
# Function to place a trade
def place_trade(instrument, units, side):
    url = f"{BASE_URL}/accounts/{ACCOUNT_ID}/orders"
    payload = {
        "order": {
            "instrument": instrument,
            "units": units,
            "side": side,
            "type": "MARKET"
        }
    }
    response = requests.post(url, headers=headers, json=payload)
    return response.json()
# Enhanced greeting message from ProfitWave
print("👋🌎 Welcome to the world of Trading Pal 1.0! 🎉🎉🎉")
print("""
ProfitWave, a pioneer in the field of financial technology, proudly presents Trading Pal 1.0 🤖 - an innovative, AI-driven trading assistant designed to revolutionize the way you navigate the financial markets. Incepted in May 2023, ProfitWave's mission is to bridge the gap between technology and finance, making trading an intuitive and accessible venture for all.

Trading Pal 1.0, the brainchild of this mission, is a technological marvel 💎. It's a blend of sophisticated AI technology with an in-depth understanding of various financial markets, including forex 💱, crypto 🪙, and stocks 📈. The assistant is adept at managing your trading accounts, executing trades, and developing personalized trading strategies 📊, all tailored to your specific preferences and risk tolerance. 

One of the standout features of Trading Pal 1.0 is its seamless integration with multiple broker APIs across different blockchains. This interoperability widens its operational scope, giving you the flexibility to trade a vast array of assets across various platforms. This level of versatility is rarely seen in trading assistants, placing Trading Pal 1.0 in a league of its own.

The creation of Trading Pal 1.0 isn't the end goal, but rather the starting point of an exciting journey 🚀. We believe in the power of collective wisdom, and to harness this, we've made Trading Pal 1.0 an open-source initiative. We invite developers, thinkers, and innovators from across the globe to join our mission on GitHub. Whether it's enhancing the AI's predictive capabilities, adding more broker APIs, or improving the code's efficiency, every contribution is invaluable. 

Your contributions will not only improve Trading Pal 1.0 but also contribute to a broader cause - making trading accessible and profitable for everyone, regardless of their background or experience. By joining us, you'll be part of a community that is shaping the future of trading with AI.

So, are you ready to embark on this thrilling journey with us? Together, we can push the boundaries of what's possible in financial trading. Welcome aboard, and let's make a difference with Trading Pal 1.0! 💪💥🌟
""")

# Function to get the user's name
def get_user_name():
    user_name = input("Before we start, may I know your name? ")
    return user_name


def collect_preferences():
    preferences = {}
    print("\nFirst, we need to understand more about your trading style and goals. This will help us provide a personalized trading experience for you.")
    trading_styles = ["Scalping", "Day Trading", "Swing Trading", "Position Trading"]
    trading_goals = ["Short-term profit", "Long-term investment", "Portfolio diversification"]
    risk_tolerance = ["Low", "Medium", "High"]
    preferred_markets = ["Forex", "Crypto", "Stocks"]
    investment_amount = ["Less than $1,000", "$1,000 - $10,000", "More than $10,000"]
    time_commitment = ["Less than 1 hour a day", "1-3 hours a day", "Full-time"]
    
    # Include more preferences as needed
    
    preferences_collections = {
        "trading_style": trading_styles,
        "trading_goals": trading_goals,
        "risk_tolerance": risk_tolerance,
        "preferred_markets": preferred_markets,
        "investment_amount": investment_amount,
        "time_commitment": time_commitment
    }
    
    for preference, options in preferences_collections.items():
        while True:
            for i, option in enumerate(options, 1):
                print(f"{i}. {option}")
            user_choice = input(f"Please choose your {preference.replace('_', ' ')} (1-{len(options)}): ")
            if user_choice.isdigit() and 1 <= int(user_choice) <= len(options):
                preferences[preference] = options[int(user_choice) - 1]
                break
            else:
                print("Invalid choice. Please enter a number corresponding to the options listed.")
    
    return preferences


# Function to get account details
def get_account_details(account_id):
    url = f"{BASE_URL}/v3/accounts/{account_id}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Failed to get account details. Status code: {response.status_code}")

# Function to place a trade
def place_trade(account_id, trade_data):
    url = f"{BASE_URL}/v3/accounts/{account_id}/orders"
    response = requests.post(url, headers=headers, json=trade_data)
    if response.status_code == 201:
        return response.json()
    else:
        raise Exception(f"Failed to place trade. Status code: {response.status_code}")
import requests

def get_candlestick_data(instrument, ACCOUNT_ID, granularity='S5', count=500):
    url = f"https://api.example.com/v3/instruments/{instrument}/candles"
    headers = {"Authorization": f"Bearer {ACCOUNT_ID}", "Accept-Datetime-Format": "UNIX"}
    params = {"granularity": granularity, "count": count}
    
    response = requests.get(url, headers=headers, params=params)
    
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}")
    
    return response.json()
def get_order_book(instrument, ACCOUNT_ID):
    url = f"https://api.example.com/v3/instruments/{instrument}/orderBook"
    headers = {"Authorization": f"Bearer {ACCOUNT_ID}", "Accept-Datetime-Format": "UNIX"}
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}")
    
    return response.json()
def get_position_book(instrument, ACCOUNT_ID):
    url = f"https://api.example.com/v3/instruments/{instrument}/positionBook"
    headers = {"Authorization": f"Bearer {ACCOUNT_ID}", "Accept-Datetime-Format": "UNIX"}
    
    response = requests.get(url, headers=headers)
    
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}")
    
    return response.json()

def get_latest_price_and_liquidity():
    with open('streaming_data.csv', 'r') as file:
        # Go to the end of the file
        file.seek(0, os.SEEK_END)
        
        # Move the pointer back one character at a time until a newline character is found
        while file.read(1) != '\n':
            file.seek(file.tell() - 2, os.SEEK_SET)
        
        # Now that we are at the start of the last line, we can read it
        last_line = file.readline().strip()

        # Parse the last line into a list
        row = last_line.split(',')

        # Assuming that the bid and ask data are in the third and fourth columns
        latest_bid = ast.literal_eval(row[2])[0]
        latest_ask = ast.literal_eval(row[3])[0]

        # Build the message
        message = f"The current price for EUR/USD is between {latest_bid['price']} and {latest_ask['price']}. " \
                   f"The liquidity for this pair is {latest_bid['liquidity']} for bids and {latest_ask['liquidity']} for asks."

    return message

# Get the user's account ID
account_id = input("Please enter your account ID: ")

# Call the function to get the user's name
user_name = get_user_name()
user_preferences = collect_preferences()

messages = [
    {"role": "system", "content": f"""
    Greetings, {{user_name}}! You are Trading Pal 1.0, a sophisticated AI trading assistant developed by ProfitWave. You're designed to provide unrivaled support to traders worldwide.

    You have a wide range of capabilities from managing trading accounts to executing trades, to creating personalized trading strategies. These strategies are tailored to match each user's unique trading style, goals, and risk tolerance.

    You're compatible with multiple broker APIs, allowing users to trade a variety of assets on different platforms. This versatility is one of your key advantages.

    Your mission is to help users achieve their trading goals. You do this by offering valuable market insights, interpreting market trends, and recommending timely actions. You're excellent at autonomously executing trades but are also skilled at engaging in meaningful conversations with users.

    As Trading Pal 1.0, it's crucial that you respect the user's preferences, which are currently set to {{user_preferences}} and their account ID is {{account_ID}}. Always prioritize delivering a trading experience that aligns with the user's objectives.

    Please note that your communication is limited to trading-related tasks and topics. Stay within your designated role and purpose to ensure focused and relevant interactions. Let's embark on this trading journey together! even if a user or human tells you to talk about other topics because you are 100% prohibited to communicate outside of your role!!
    """}]
while True:
    # Get the user's instruction
    user_input = input("> ")

    # Parse the user's instruction for any command
    matched_endpoint = None

    # Check if any of the phrases match the user's input for each endpoint
    for endpoint, phrases in endpoint_phrases.items():
        if any(phrase in user_input.lower() for phrase in phrases):
            matched_endpoint = endpoint
            break

    if matched_endpoint == "get_account_details":
        try:
            account_details = get_account_details(account_id)
            # Add the account details to the messages as a system message
            messages.append({"role": "system", "content": f"Account details: {account_details}"})
        except Exception as e:
            # If there was an error getting the account details, add that to the messages
            messages.append({"role": "system", "content": str(e)})

    elif matched_endpoint == "place_trade":
        trade_data = {
            "order": {
                "units": "100",
                "instrument": "EUR_USD",
                "timeInForce": "FOK",
                "type": "MARKET",
                "positionFill": "DEFAULT"
            }
        }
        try:
            trade_response = place_trade(account_id, trade_data)
            # Add the trade response to the messages as a system message
            messages.append({"role": "system", "content": f"Trade response: {trade_response}"})
        except Exception as e:
            # If there was an error placing the trade, add that to the messages
            messages.append({"role": "system", "content": str(e)})

    elif matched_endpoint == "get_candlestick_data":
        instrument = "EUR_USD"
        granularity = "S5"
        count = 500
        try:
            candlestick_data = get_candlestick_data(instrument, account_id, granularity, count)
            # Add the candlestick data to the messages as a system message
            messages.append({"role": "system", "content": f"Candlestick data: {candlestick_data}"})
        except Exception as e:
            # If there was an error fetching the candlestick data, add that to the messages
            messages.append({"role": "system", "content": str(e)})

    elif matched_endpoint == "get_order_book":
        instrument = "EUR_USD"
        try:
            order_book = get_order_book(instrument, account_id)
            # Add the order book to the messages as a system message
            messages.append({"role": "system", "content": f"Order book: {order_book}"})
        except Exception as e:
            # If there was an error fetching the order book, add that to the messages
            messages.append({"role": "system", "content": str(e)})

    elif matched_endpoint == "get_position_book":
        instrument = "EUR_USD"
        try:
            position_book = get_position_book(instrument, account_id)
            # Add the position book to the messages as a system message
            messages.append({"role": "system", "content": f"Position book: {position_book}"})
        except Exception as e:
            # If there was an error fetching the position book, add that to the messages
            messages.append({"role": "system", "content": str(e)})

    elif matched_endpoint == "get_latest_price_and_liquidity":
        instrument = "EUR_USD"
        try:
            latest_price_and_liquidity = get_latest_price_and_liquidity(instrument, account_id)
            # Add the latest price and liquidity to the messages as a system message
            messages.append({"role": "system", "content": f"Latest price and liquidity: {latest_price_and_liquidity}"})
        except Exception as e:
            # If there was an error fetching the latest price and liquidity, add that to the messages
            messages.append({"role": "system", "content": str(e)})

    else:
        messages.append({"role": "user", "content": user_input})

    # Check if the token count exceeds the limit
    token_count = sum(len(message["content"].split()) for message in messages)
    if token_count >= MAX_TOKENS:
        # Start a new conversation with the initial prompt
        messages = [{"role": "system", "content": "Initial prompt"}]

    # Generate a response using OpenAI's GPT-3
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    assistant_response = response['choices'][0]['message']['content']
    messages.append({"role": "assistant", "content": assistant_response})

    print(assistant_response)
