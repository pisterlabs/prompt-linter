import re
import os
import openai
import requests
import json
from time import sleep
from raven_functions import *
from solr_functions import *
import urllib3


default_sleep = 1
urllib3.disable_warnings()
open_ai_api_key = read_file('openaiapikey.txt')
openai.api_key = open_ai_api_key
last_msg = {'time':0.0}
#default_engine = 'davinci'
default_engine = 'curie-instruct-beta'


def query_gpt3(prompt):
    response = openai.Completion.create(
        engine=default_engine,
        prompt=prompt,
        temperature=0.5,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0.3,
        presence_penalty=0.3,
        stop=['PASSAGE:', 'QUERIES:', 'INSTRUCTIONS:', '<<END>>'])
    return response['choices'][0]['text']


def post_articles(articles, context):
    for article in articles:
        try:
            # TODO massage article
            payload = dict()
            #print('POST ARTICLE:', article['title'])
            payload['msg'] = str(article['title']) + ' : ' + str(article['text'])
            payload['irt'] = context['mid']
            payload['ctx'] = context['mid']
            payload['key'] = 'encyclopedia.article'
            payload['sid'] = 'encyclopedia.wiki'
            result = nexus_post(payload)
            #print(result)
        except Exception as oops:
            print('ERROR in ENCYCLOPEDIA/POST_ARTICLES:', oops)


def get_search_queries(text):
    prompt = read_file('prompt_search_query.txt')
    prompt = prompt.replace('<<PASSAGE>>', text)
    results = query_gpt3(prompt).split(',')
    return [i.strip() for i in results]


def query_nexus():
    global last_msg
    try:
        stream = nexus_get(key='context', start=last_msg['time'])
        for context in stream:
            if context['time'] <= last_msg['time']:
                continue
            if context['time'] > last_msg['time']:
                last_msg = context
            queries = get_search_queries(context['msg'])
            #print('QUERIES:', queries)
            articles = list()
            for query in queries:
                result = solr_search(query)
                articles += result['response']['docs']
            post_articles(articles, context)
    except Exception as oops:
        print('ERROR in ENCYCLOPEDIA/QUERY_NEXUS:', oops)


if __name__ == '__main__':
    print('Starting Encyclopedia Service')
    while True:
        query_nexus()
        sleep(default_sleep)
