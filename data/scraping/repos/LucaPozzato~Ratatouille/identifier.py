import requests
import json
from serpapi import GoogleSearch
import openai
from datetime import date
import os

IMGBB_KEY = os.environ['IMGBB_KEY']
IMGBB_URL = 'https://api.imgbb.com/1/upload?expiration=60&key='+IMGBB_KEY

SERPAPI_KEY = os.environ['SERPAPI_KEY']

OPENAI_KEY = os.environ['OPENAI_KEY']

true = True

def get_product(image):
    url = upload(image=image)
    product = product_name(url=url)
    product_dict = dict_gen(product)
    return product_dict

def upload(image):
    r = requests.post(IMGBB_URL, data={'image': image})
    response = r.text
    dict = json.loads(response)
    data = dict["data"]
    IMAGE_URL = data["url"]
    return IMAGE_URL

def product_name(url):
    params = {
    "engine": "google_lens",
    "url": url,
    "api_key": SERPAPI_KEY
    }

    search = GoogleSearch(params)
    results = search.get_dict()

    if 'knowledge_graph' in results:
        return results['knowledge_graph'][0]['title']
    else:
        return results['visual_matches'][0]['title']

def dict_gen(product):
    client = openai.OpenAI(
        api_key=OPENAI_KEY,
    )

    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": "I need a dict that has a product, its product catgeory and the shelf_life and the day. The dictionary should only have as keys: product, category in english, shelf_life is just the number of days, date = 'n/a'. The product is" + product + ".I only need the array, please don't generate other text. Don't say anything else."}
        ]
    )
    return completion.choices[0].message.content
