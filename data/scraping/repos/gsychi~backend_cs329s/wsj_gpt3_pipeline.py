from scrape_wsj_articles import scrape_day
from singleshot import generate_blurb
import json
from datetime import date, timedelta, datetime
import openai
from extract_sentiment import process_article
import os

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


def create_wsj_predictions(day):
    date_str = day.strftime('%Y-%m-%d')
    file_name = f"data/wsj_predictions/{date_str}_with_price.json"

    if os.path.isfile(file_name):
        print("Predictions already generated for given date.")
        return
    else:
        """
        Construct WSJ Database
        """
        date_str = day.strftime('%Y-%m-%d')

        print("Extracting information for", date_str)
        scrape_day(day)

        file_name = f"data/wsj/{date_str}.json"

        """
        Make predictions
        """
        init_openai()
        processed_articles = []
        with open(file_name) as f:
            data = json.load(f)
            for d in data:
                try:
                    stock_input = generate_blurb(d, "data/prompts/vaccine_prompt.txt")
                    response = call_openai(stock_input)
                    d['blurb'] = stock_input
                    d['Prediction'] = response['choices'][0]['text']

                    print("Predicted article.")

                    processed_articles.append(d)
                except:
                    continue

        file_name = f"data/wsj_predictions/{date_str}.json"
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
        file_name = f"data/wsj_predictions/{date_str}_with_price.json"
        with open(file_name, 'w') as f:
            json.dump(output_info, f)


if __name__ == "__main__":
    day = datetime.today() - timedelta(days=3)
    create_wsj_predictions(day)
