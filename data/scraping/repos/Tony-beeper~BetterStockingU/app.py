import datetime
from flask import Flask
from pymongo import MongoClient
from flask_cors import CORS
import os
from dotenv import load_dotenv
import cohere
import training
from flask import request
import requests
import spacy
import pandas as pd
import itertools
import json
from urllib.parse import unquote

# Load English tokenizer, tagger, parser and NER
nlp = spacy.load("en_core_web_lg")

load_dotenv()
COHERE = os.getenv('COHERE')
MONGO_USER = os.getenv('MONGO_USER')
MONGO_PASS = os.getenv('MONGO_PASS')
TWITTER_BRARER = os.getenv('bearer_token')
MAX_RESULTS = 10
print(
    f'mongodb+srv://{MONGO_USER}:{MONGO_PASS}@cluster0.nuben.mongodb.net/?retryWrites=true&w=majority')
client = MongoClient(
    f'mongodb+srv://{MONGO_USER}:{MONGO_PASS}@cluster0.nuben.mongodb.net/?retryWrites=true&w=majority')
db = client.stockingu
co = cohere.Client(COHERE)

app = Flask(__name__)
CORS(app)
print(db)


def get_rating(company, posts):
    collection = db.stockingu
    company_rating = collection.find_one({'data.company': company})
    # if company_rating:
    #     print("cache hit")
    #     company_rating['_id'] = str(company_rating['_id'])
    #     return company_rating
    # else:
    classifications = co.classify(
        model='finance-sentiment',
        inputs=posts,
    )
    output = []
    average = 0
    num_posts = 0
    for cl in classifications.classifications:
        print(cl)
        title = cl.input
        sentiment = cl.prediction
        confidence_p = cl.labels['POSITIVE'].confidence
        confidence_n = cl.labels['NEGATIVE'].confidence
        confidence = 0
        if sentiment == "POSITIVE":
            confidence = confidence_p+1
            num_posts += 1
        elif sentiment == "NEGATIVE":
            confidence = 1 + (confidence_n*-1)
            num_posts += 1

        output.append({
            'title': title,
            'sentiment': sentiment,
            'confidence': confidence
        })
        average += confidence
    rating = average / num_posts
    timestamp = datetime.datetime.utcnow()
    result = {'data': {'company': company,
                       'rating': rating, 'timestamp': timestamp}}
    collection.insert_one(result)
    result['_id'] = str(result['_id'])
    return {'data': output, 'average': result}


@app.route("/api", methods=['POST'])
def api():
    if request.method == 'POST':
        posts = request.json['posts']
        company = request.json['company']
        print({"you entered ": posts})
        return get_rating(company, posts)

    return {"hello": "world"}


def get_twitter_page_data(query, next_token):
    # api-endpoint
    URL = "https://api.twitter.com/2/tweets/search/recent?query={}&max_results={}".format(
        query, MAX_RESULTS)
    nest_token_str = ("&next_token=" + next_token) if next_token else ""
    URL = URL + nest_token_str

    headers = {"Authorization": TWITTER_BRARER}

    # sending get request and saving the response as response object
    r = requests.get(url=URL, headers=headers)

    # extracting data in json format
    return r.json()


def get_twitter_data(query):
    round = 0
    next_token = ""
    text_arr = []
    while (round < 10):
        result = get_twitter_page_data(query=query, next_token=next_token)
        if (not 'data' in result):
            break
        for tweet in result['data']:
            text_arr.append(tweet['text'])
        if ('next_token' in result['meta']):
            next_token = result['meta']['next_token']
        round = round + 1
    return text_arr


@app.route("/api/twitter/search/<company>", methods=['GET'])
def search_twitter(company):
    # company = unquote(company)
    query = "lang%3Aen%20{}".format(company)
    print(query)
    text_arr = get_twitter_data(query)

    text_arr = list(dict.fromkeys(text_arr))
    if (not len(text_arr)):
        return 'bad request!', 400
    all_stopwords = nlp.Defaults.stop_words

    all_stopwords.add('rt')
    all_stopwords.add('#')
    all_stopwords.add('amp')
    all_stopwords.add(unquote(company.lower()))

    text = ' '.join(text_arr).lower().replace('#', '')
    doc = nlp(text)
    org_arr = [
        chunk.text for chunk in doc.noun_chunks if chunk.text not in all_stopwords]
    # org_arr = [t.text for t in doc.ents if t.label_ == "PRODUCT"]
    df = pd.Index(org_arr)
    re = df.value_counts().to_dict()
    return {"cohere": get_rating(unquote(company), text_arr), "frequency": json.dumps(dict(itertools.islice(re.items(), 3)))}


@app.route("/api/twitter/user/<username>", methods=['GET'])
def search_by_user(username):
    query = "lang%3Aen%20from%3A{}".format(username)
    text_arr = get_twitter_data(query)
    return {"text_arr": text_arr}


@app.route("/api/wordfrequency/<ignore_word>", methods=['GET'])
def find_frequency(ignore_word):
    text_arr = request.json['text_arr']
    text_arr = list(dict.fromkeys(text_arr))
    all_stopwords = nlp.Defaults.stop_words

    all_stopwords.add('rt')
    all_stopwords.add('#')
    all_stopwords.add(unquote(ignore_word.lower()))

    text = ' '.join(text_arr).lower().replace('#', '')
    doc = nlp(text)
    org_arr = [
        chunk.text for chunk in doc.noun_chunks if chunk.text not in all_stopwords]
    # org_arr = [t.text for t in doc.ents if t.label_ == "PRODUCT"]
    df = pd.Index(org_arr)
    re = df.value_counts().to_dict()
    return {"data": json.dumps(dict(itertools.islice(re.items(), 3)))}


@app.route("/test", methods=['GET'])
def test():
    return {"text_arr": "a"}
