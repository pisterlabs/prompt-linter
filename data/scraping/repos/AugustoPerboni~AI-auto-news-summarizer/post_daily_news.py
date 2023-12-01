# Python libraries
import requests
import re
import datetime
import json
import requests
from datetime import datetime

# Third party libraries
import feedparser
import pandas as pd
import openai
from bs4 import BeautifulSoup
import facebook as fb

# Microsoft account paid api
openai.api_key = 'ADD YOUR API KEY HERE'

# Chat GPT interaction
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

def scrape_rss_feed(rss_url):
    '''
    Get all the news from the feed and return a list of dictionaries with the news informations
    '''
    feed = feedparser.parse(rss_url)
    feed_data = []

    for entry in feed.entries:
        news = {}
        news['title'] = entry.title
        # news['publication_date'] = entry.published
        # news['structure_publication_date'] = entry.published_parsed
        news['summary'] = entry.summary
        news['link'] = entry.link

        feed_data.append(news)

    return feed_data

def generate_summaries_str(feed_data, numarize=True):
    '''
    Get the summary of each news and create a string with were each line is one summary with information inside<>
    '''

    summaries = str()
    if numarize:
        for index in range(len(feed_data)):
            summaries += f'\n <{index}:' + feed_data[index]['summary'] + '>'
    else:
        for index in range(len(feed_data)):
            summaries += f'\n <' + feed_data[index]['summary'] + '>'

    return summaries

def generate_filter_summary_prompt(bbc_feed_str, number_of_news):
    '''
    Generates the prompt that will be used to choose the best news 
    '''
    filter_summaries_prompt = f"""
You are a tool that helps a team from a journal to find the news with more audience potential\
from a list of summaries. 
        
You will receive a list of summaries delimited by ##, each summary follows the following pattern:
<index:summary_content>.

You respond with the index of the {str(number_of_news)} elements with more potential in a json with one key\
called indexes with one list inside with all the choose indexes
    
Consider that the summaries are in random order. You will
define the order of importance.
    
Do not make any selection without analyzing all.
    
Answer the chosen ones, one after another inside <>.
Do not number them.
        
Summaries: #{bbc_feed_str}#
    """
    return filter_summaries_prompt

def generate_text_prompt(news_formated):
    generate_text_prompt = f"""
You are a journalist in charge to write a text with a small description of the day's most important events.
All today's events are provided and delimited by ## and each one follows this pattern: <title:news_text_content>.

Perform the following actions: 

1 - Write one paragraph of at least 60 words summing up each piece of news, and start introducing the situation before the real sum-up.
2 - Create one single and continuous divided paragraphs narrative with the previously written text

In the final text:
- Each news deserve a paragraph of attention not less and not more
- Do not write introduction or conclusion paragraphs

Write the output in the following format:
< Paragraph with the 1 news content>
< Paragraph with the 2 news content>
...
< Paragraph with the last news content>

List of news: #{news_formated}#
    
    """
    return generate_text_prompt

def scrape_one_news(news_ulr):
    response = requests.get(news_ulr)
    if not response.ok:
        return None
    else:
        soup = BeautifulSoup(response.content, features='lxml')
        text_objects = soup.find_all(attrs={"data-component":"text-block"})
        news_text = '\n'.join([element.text for element in text_objects])

        return news_text



if __name__ == '__main__':

    # RSS Link
    bbc_rss_feed_url = "https://feeds.bbci.co.uk/news/rss.xml"

    # Scrape the feed from BBC-UK
    bbc_feed_data = scrape_rss_feed(bbc_rss_feed_url)

    # Generate a string with all the summaries in a specific pattern
    summaries_string_formated = generate_summaries_str(bbc_feed_data)

    # Create a prompt to choose the best summaries from the summaries string
    filter_summaries_prompt = generate_filter_summary_prompt(summaries_string_formated, 5)

    # Recieve the answer in a json 
    bbc_selected_news_dic = json.loads(get_completion(filter_summaries_prompt))
    bbc_selected_news_indexes = bbc_selected_news_dic['indexes']

    # Scrape the news and create a formated string to be used in chat gpt
    news_full_content_formated = []
    for index in bbc_selected_news_indexes:
        news_text = scrape_one_news(bbc_feed_data[index]['link'])
        if news_text != None:
            news_formated_string = '<' + bbc_feed_data[index]['title'] + ':' + news_text + '>'
            news_full_content_formated.append(news_formated_string)
    
    # Write the prompt to later generate the final text
    text_prompt = generate_text_prompt(news_full_content_formated)
    # Generate the final text
    final_text = get_completion(text_prompt, model='gpt-3.5-turbo-16k')
    
    access_token =  'FILL WITH YOURS'
    page_id = 'FILL WITH YOURS'

    fb_graph = fb.GraphAPI(access_token)
    post_id = fb_graph.put_object(parent_object=page_id, connection_name='feed', message=final_text)

    # Daily post status

    # Format the date and time
    now = datetime.now()
    formatted_now = now.strftime("%Y-%m-%d %H:%M:%S")
    print('Day ', formatted_now, ' status:')
    if 'id' in post_id:
        print("    -> Post was successful. ID of the new post is:\n", post_id['id'])
    else:
        print("    -> There was an error while posting\n.")
