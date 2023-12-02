import nltk
import openai
import requests
import pandas as pd
from pytrends.request import TrendReq
from google.oauth2 import service_account
from googleapiclient.discovery import build
from config import Config
from textstat import textstat
from collections import Counter
from api.middleware.error_handlers import internal_error_handler

#Only ONE TIME Download is required
#nltk.download('punkt')

openai.api_key = Config.OPENAI_API_KEY


# Replace this with the path to your service account key JSON file
KEY_FILE_LOCATION = '/home/ubuntu/development/Backend/dronacharya.json'
VIEW_ID = 'YOUR_VIEW_ID'  # Replace this with your Google Analytics View ID

# Set up the Google Analytics API client
credentials = service_account.Credentials.from_service_account_file(KEY_FILE_LOCATION)
analytics = build('analyticsreporting', 'v4', credentials=credentials)

@internal_error_handler
def generate_content_ideas(prompt, num_ideas=5, format='content idea', examples=None, constraints=None):
    prompt_intro = f"Generate {num_ideas} {format}s for the following topic: {prompt}"
    
    if examples:
        prompt_intro += f"\nExamples:\n{examples}"
    
    if constraints:
        prompt_intro += f"\nConstraints:\n{constraints}"
    
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt_intro,
        temperature=0.7,
        max_tokens=100,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=num_ideas,
    )

    ideas = [choice.text.strip() for choice in response.choices]
    return {
        "data": ideas
    }

@internal_error_handler
def generate_content_brief(keywords, target_audience, competitors, num_sections=5):
    prompt = f"Create a content brief for an article targeting the keywords {', '.join(keywords)}, with insights on the target audience: {target_audience}, and considering the competitors: {', '.join(competitors)}. Include {num_sections} sections in the outline."

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    content_brief = response.choices[0].text.strip()
    return {
        "brief": content_brief,
    }

@internal_error_handler
def generate_social_media_posts(topic, num_posts=3):
    prompt = f"Generate {num_posts} engaging social media posts for the topic '{topic}', including captions and hashtags."

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=200,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=num_posts,
    )

    posts = [{'caption': choice.text.strip().split('\n')[0], 'hashtags': choice.text.strip().split('\n')[1]} for choice in response.choices]
    return {
        "posts": posts,
    }


@internal_error_handler
def seo_optimization(content):
    # Tokenize content into words
    words = nltk.word_tokenize(content)

    # Calculate keyword density
    word_count = len(words)
    word_frequencies = Counter(words)
    keyword_density = {word: (count / word_count) * 100 for word, count in word_frequencies.items()}

    # Analyze readability
    readability = {
        'flesch_reading_ease': textstat.flesch_reading_ease(content),
        'smog_index': textstat.smog_index(content),
        'flesch_kincaid_grade': textstat.flesch_kincaid_grade(content),
        'coleman_liau_index': textstat.coleman_liau_index(content),
        'automated_readability_index': textstat.automated_readability_index(content),
        'dale_chall_readability_score': textstat.dale_chall_readability_score(content),
        'difficult_words': textstat.difficult_words(content),
        'linsear_write_formula': textstat.linsear_write_formula(content),
        'gunning_fog': textstat.gunning_fog(content),
    }

    return {
        'keyword_density': keyword_density,
        'readability': readability
    }

@internal_error_handler
def get_trending_topics(query):
    pytrends = TrendReq(hl='en-US', tz=360)
    kw_list = [query]
    pytrends.build_payload(kw_list, cat=0, timeframe='now 7-d', geo='', gprop='')
    related_queries = pytrends.related_queries()
    top_related_queries = related_queries[kw_list[0]]['top']
    trending_topics = top_related_queries['query'].tolist()

    return {
        "response": trending_topics,
    }

@internal_error_handler
def get_keyword_suggestions(query, api_key):
    url = f"https://api.semrush.com/?type=phrase_related&phrase={query}&key={api_key}&database=us"
    response = requests.get(url)
    keyword_data = response.text.split('\r\n')[1:-1]  # Skipping the first line (header) and last line (empty)
    keywords = [line.split(';')[0] for line in keyword_data]
    return keywords

@internal_error_handler
def repurpose_content(content, target_format):
    prompt = f"Repurpose the following content into a {target_format}:\n\n{content}\n\n---\n\n{target_format} Content:"

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0.7,
        max_tokens=300,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    )

    repurposed_content = response.choices[0].text.strip()
    return {
        "repurposed" : repurposed_content
    }


def get_analytics_data(start_date, end_date):
    response = analytics.reports().batchGet(
        body={
            'reportRequests': [
                {
                    'viewId': VIEW_ID,
                    'dateRanges': [{'startDate': start_date, 'endDate': end_date}],
                    'metrics': [{'expression': 'ga:sessions'}, {'expression': 'ga:pageviews'}],
                    'dimensions': [{'name': 'ga:pageTitle'}, {'name': 'ga:date'}],
                    'orderBys': [{'fieldName': 'ga:date', 'sortOrder': 'ASCENDING'}]
                }]
        }
    ).execute()
    return response

def parse_response(response):
    report = response.get('reports', [])[0]
    columnHeader = report.get('columnHeader', {})
    dimensionHeaders = columnHeader.get('dimensions', [])
    metricHeaders = columnHeader.get('metricHeader', {}).get('metricHeaderEntries', [])

    data = report.get('data', {})
    rows = data.get('rows', [])

    processed_data = []

    for row in rows:
        dimensions = row.get('dimensions', [])
        metrics = row.get('metrics', [])[0].get('values', [])

        row_data = {}
        for header, dimension in zip(dimensionHeaders, dimensions):
            row_data[header] = dimension

        for header, metric in zip(metricHeaders, metrics):
            header_name = header.get('name', '')
            row_data[header_name] = metric

        processed_data.append(row_data)

    df = pd.DataFrame(processed_data)
    return df



