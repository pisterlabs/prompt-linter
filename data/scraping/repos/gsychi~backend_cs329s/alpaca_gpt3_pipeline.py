from extract_sentiment import process_article
from scrape_utils import scrape_articles_alpaca
import json
from datetime import date, timedelta, datetime
import openai
import os

def generate_blurb(article, base):
    #read from data/prompts/flight_prompt.txt
    with open(base) as f:
        prompt = f.read()
        prompt = prompt.replace("{TITLE}", article['title'])
        prompt = prompt.replace("{SUMMARY}", article['summary'])
        prompt = prompt.replace("{STOCKS}", ", ".join(article['tickers']))
        return prompt

def init_openai():
    openai.organization = 'org-JIX8loMHNMi4CMFfJJsPYva3'
    openai.api_key = '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'
    openai.Engine.list()

def call_openai(blurb):
    openai.Engine.retrieve("text-davinci-001")
    response = openai.Completion.create(
        engine="text-davinci-001",
        prompt=blurb,
        max_tokens=128,  # previously 64, 0
        temperature=.2)

    return response

def create_alpaca_predictions(day):
    date_str = day.strftime('%Y-%m-%d')
    file_name = f"data/alpaca_predictions/{date_str}_with_price.json"

    if os.path.isfile(file_name):
        print("Predictions already generated for given date.")
        return
    else:
        """
        Construct Alpaca Database
        """
        print("Extracting information for", date_str)

        file_name = f"data/alpaca/{date_str}.json"
        # if the file exists, skip
        if os.path.isfile(file_name):
            with open(file_name) as f:
                print("Day scraped with", len(json.load(f)), "articles.")
        else:
            articles = scrape_articles_alpaca(date_str)
            with open(file_name, "w") as f:
                json.dump(articles, f)

        """
        Construct Blurb + Prediction json
        """
        init_openai()

        processed_articles = []
        with open(file_name) as f:
            data = json.load(f)
            for d in data:
                stock_input = generate_blurb(d, "data/prompts/vaccine_prompt_alpaca.txt")
                response = call_openai(stock_input)
                d['blurb'] = stock_input
                d['Prediction'] = response['choices'][0]['text']

                print("Predicted article.")

                processed_articles.append(d)

        file_name = f"data/alpaca_predictions/{date_str}.json"
        with open(file_name, "w") as f:
            json.dump(processed_articles, f)

        """
        Create with Sentiments
        """
        with open(file_name) as f:
            data = json.load(f)

        output_info = []
        for article in data:
            output_info.append(process_article(article))

        # save output_info to a json file
        file_name = f"data/alpaca_predictions/{date_str}_with_price.json"
        with open(file_name, 'w') as f:
            json.dump(output_info, f)

if __name__ == "__main__":
    day = datetime.today() - timedelta(days=3)
    create_alpaca_predictions(day)