import openai
import requests
import json
import re
from statistics import mean
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Now you can access the keys
openai.api_key = os.getenv('OPENAI_API_KEY')

def get_eia_data(start_year, end_year):
    api_key = os.getenv('EIA_API_KEY')
    base_url = "https://api.eia.gov/v2/petroleum/pri/dfp1/data/"
    url = f"{base_url}?api_key={api_key}&frequency=monthly&data[0]=value&start={start_year}-01&end={end_year}-12&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000"

    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        # Check if data exists for the given period
        if not data or not data['response']['data']:
            print(f"No data available for the period {start_year}-{end_year}")
            return None
        else:
            return data
    else:
        print(f"Unable to retrieve data. Error: {response.status_code}")
        return None

def calculate_average_price(data):
    # Extract the 'value' field from each data point and ignore data points without a value
    prices = [float(item['value']) for item in data['response']['data'] if 'value' in item and item['value'] is not None]
    # Return the average price
    return mean(prices)

def ask_gpt(average_price, question):
    gpt_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "assistant", "content": "The average price of crude oil is currently at " + str(average_price) + " dollars per barrel."},
            {"role": "user", "content": question}
        ]
    )
    return gpt_response.choices[0].message['content']

def parse_years_from_question(question):
    # Extract years from the question. This function assumes years are four-digit numbers.
    years = re.findall(r"\b\d{4}\b", question)
    years = [int(year) for year in years]
    if len(years) >= 2:
        return min(years), max(years)
    else:
        print("Please provide two years in your question to define the period.")
        return None, None

def main():
    question = input("What would you like to ask about the oil market? ")
    start_year, end_year = parse_years_from_question(question)

    if start_year is not None and end_year is not None:
        data = get_eia_data(start_year, end_year)
        if data is not None:
            average_price = calculate_average_price(data)
            gpt_answer = ask_gpt(average_price, question)
            print(gpt_answer)
        else:
            print("Unable to continue due to error retrieving EIA data.")
    else:
        print("Unable to proceed without a valid time period.")

if __name__ == "__main__":
    main()
