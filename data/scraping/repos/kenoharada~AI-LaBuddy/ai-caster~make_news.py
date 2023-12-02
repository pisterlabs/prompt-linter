import openai
import os
import glob
import json
import requests
from typing import List
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

MODEL_NAME = 'gpt-3.5-turbo-16k'
EMBEDDING_MODEL_NAME = 'text-embedding-ada-002'

import requests
from bs4 import BeautifulSoup

def fetch_texts_from_url_list(url_list: List[str]):
    """
    Return a dictionary of URLs and fetched texts.
    """
    result_dict = dict()
    for url in url_list:
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(f"Request failed: {response.text}")

        soup = BeautifulSoup(response.text, 'html.parser')
        body = soup.find('body')
        if body is not None:
            text = body.get_text()
            text = text.strip()
            decoded_text = text.encode(response.encoding).decode('utf-8', 'ignore').strip()
            result_dict[url] = decoded_text
        else:
            result_dict[url] = ''
    return result_dict


def process_tweet(tweet):
    messages = [
            {"role": "system", "content": "You are a newscaster with credibility and clear, concise commentary. Your goal is to create a script to be read on the news. The first step is to read the twitter tweet text surrounded by #### which is the source of the news. If the tweet contains URLs and you need to refer to them for clarification, refer to the text at the URL. fetch_texts_from_url_list, which takes a list of URLs as its argument, will return the URL as key and the text as value. Next, provide a brief description of the words that you think need to be explained to describe the news. Finally, output the text to be read to the viewer to make the news easier to understand. Use the following format: words_to_be_explained: <json with words and explanations> script: <the text to be read in news>"},
            {"role": "user", "content": f"tweet: #####{tweet}#####"},
            ]
    response = openai.ChatCompletion.create(
        model=MODEL_NAME,
        messages=messages,
        functions=[
            {
                "name": "fetch_texts_from_url_list",
                "description": "return dict of url and fetched text",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "url_list": {
                            "type": "array",
                            "items": {
                                "type": "string"
                            },
                            "description": "urls to fetch text from",
                        },
                    },
                    "required": ["url_list"],
                },
            },
        ],
    )
    message = response["choices"][0]["message"]
    # try:
    if message.get("function_call"):
        function_name = message["function_call"]["name"]

        if function_name == "fetch_texts_from_url_list":
            if message.get("url_list") is None:
                print(message, eval(message["function_call"]["arguments"]))
                # print(parse_urls(str(message["function_call"]["arguments"])))
                
            function_response = fetch_texts_from_url_list(url_list=eval(message["function_call"]["arguments"])['url_list'])
            print(function_name, function_response)
            second_response = openai.ChatCompletion.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a newscaster with credibility and clear, concise commentary. Your goal is to create a script to be read on the news. The first step is to read the twitter tweet text surrounded by #### which is the source of the news. If the tweet contains URLs and you need to refer to them for clarification, refer to the text at the URL. fetch_texts_from_url_list, which takes a list of URLs as its argument, will return the URL as key and the text as value. Next, provide a brief description of the words that you think need to be explained to describe the news. Finally, output the text to be read to the viewer to make the news easier to understand. Use the following format: words_to_be_explained: <json with words and explanations> script: <the text to be read in news>"},
                    {"role": "user", "content": f"tweet: #####{tweet}#####"},
                    message,
                    {
                        "role": "function",
                        "name": function_name,
                        "content": str(function_response),
                    }],
                functions=[
                    {
                        "name": "fetch_texts_from_url_list",
                        "description": "return dict of url and fetched text",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "url_list": {
                                    "type": "array",
                                    "items": {
                                        "type": "string"
                                    },
                                    "description": "urls to fetch text from",
                                },
                            },
                            "required": ["url_list"],
                        },
                    },
                ],
            )
            return second_response["choices"][0]["message"]["content"]
    return message["content"]
    # except Exception as e:
    #     print(e)
    #     return ''

tweets_json = glob.glob('*_tweets.json')

for influencer_tweets_json in tweets_json:
    influencer_tweets = json.load(open(influencer_tweets_json, 'r'))
    for influencer_tweet in influencer_tweets:
        print(influencer_tweet['text'])
        response = process_tweet(influencer_tweet['text'])
        if response != '':
            script = response.split('script:')[-1].strip()
            print(script)
            messages = [
                    {"role": "system", "content": "You are a professional translator. Translate the following script to be read in news into Japanese."},
                    {"role": "user", "content": script},
                ]
            response = openai.ChatCompletion.create(
                model='gpt-3.5-turbo',
                messages=messages,
                temperature=0
            )["choices"][0]["message"]["content"]
            print(response)
            print("########")
            break
    break