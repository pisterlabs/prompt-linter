#!/usr/bin/env python3

import os
import newscatcher
import langchain
from newscatcher import Newscatcher, urls
from typing import List
from lib.newsgpt import NewsCategory

def to_topic(cat: NewsCategory) -> str:
    """
    category to topic mapping to comply with newscatcher
    """
    map = {
        NewsCategory.ALL: None,
        NewsCategory.BUSINESS: "business",
        NewsCategory.POLITICS: "politics",
        NewsCategory.SPORTS: "sport",
        NewsCategory.TECHNOLOGY: "tech"
    }
    return map[cat]

def get_news(
    cat: NewsCategory, top_k: int = 10
):
    """
    Get the news site urls for the given category
    https://github.com/kotartemiy/newscatcher
    """
    urls = []
    if cat == NewsCategory.ALL:
        # urls = newscatcher.urls(language='en')
        urls = ['nytimes.com', 'theguardian.com', 'cnn.com',
                'dailymail.co.uk', 'cnet.com', 'usatoday.com',
                'independent.co.uk', 'cnbc.com', 'wired.com']
    else:
        _topic = to_topic(cat)
        urls = newscatcher.urls(topic=_topic, language='en')
        # print("Got urls: ", urls)

    # validate urls
    urls = urls[:top_k]
    print("Top K urls: ", urls)
    return urls


def get_content(urls: List[str], category: NewsCategory, content_limit: int = 15):
    print("Getting content for urls: ", urls)
    _method = 1
    content_string = ""
    for url in urls:
        nc = Newscatcher(website=url, topic=to_topic(category))
        print("\n-------------Got headlines for url: ", url)

        if nc.get_headlines() is None:
            print("No headlines found for url: ", url)
            continue
        
        # get only headlines
        if _method == 0:
            for index, headline in enumerate(nc.get_headlines()):
                print(index, headline)
                
        # get only headlines and short descs
        elif _method == 1:
            results = nc.get_news()
            articles = results['articles']
            
            print("Got articles: ", len(articles))
            if len(articles) > content_limit:
                articles = articles[:content_limit]

            for article in articles:   
                _summary = ""
                if 'summary' in article:
                    _summary = article['summary']
                    
                output = f"{article['title']} | {_summary}"
                # print(output)
                content_string += output

    if len(content_string) > 10000:
        return content_string[:10000]
    return content_string
