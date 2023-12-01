import openai
from src.auth import openai_api_key
from src.data_finder import get_data
from src.utils import get_date
from src.heatmap_downloader import download_image
from src.utils import letter_instructions
from src.logger import logging

openai.api_key = openai_api_key

def create_letter():
    with open("dags//src//market_status.txt", "r") as file:
        market_status = file.read()
    if market_status == "open":
        download_image()
        logging.info("Image downloaded")
        date = get_date()
        get_data()
        logging.info("Articles Ingested and Processed")

        with open(f"dags//src//news_letters//{date}//news_letter.txt", "r") as file:
            data = file.read()


        with open(f"dags//src//news_letters//{date}//news_letter_final.txt", "w") as file:
            article = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Turn the following data into a news letter. data: " + data + "\n" + f"use the following instructions {letter_instructions}"}])
            file.write(article.choices[0].message.content)
        logging.info("News Letter Created")
