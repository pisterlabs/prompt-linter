import openai
import requests
import json
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
        return data
    else:
        print("Error retrieving data from EIA API. Check your API key and series ID.")
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

def main():
    data = get_eia_data(1980, 1989)  # adjust the years as needed
    if data is None:
        print("Unable to continue due to error retrieving EIA data.")
    else:
        average_price = calculate_average_price(data)
        question = input("What would you like to ask about the oil market? ")
        gpt_answer = ask_gpt(average_price, question)
        print(gpt_answer)

if __name__ == "__main__":
    main()
