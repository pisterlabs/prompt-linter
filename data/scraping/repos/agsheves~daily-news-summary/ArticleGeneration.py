import anvil.users
import anvil.tables as tables
import anvil.tables.query as q
from anvil.tables import app_tables
import anvil.google.auth, anvil.google.drive, anvil.google.mail
from anvil.google.drive import app_files
import anvil.email
import anvil.server
import requests
import json
from datetime import datetime, timedelta
import anvil.secrets
import openai
import pandas as pd
from pandas import to_datetime
import re

def remove_excessive_br_tags(html_content):
    # Replace occurrences of multiple <br> (case-insensitive) with a single <br>
    cleaned_html = re.sub(r'\s*<br>\s*<hr>\s*', '<br><hr>', html_content)
    return cleaned_html


################################################
# Adds the story to the newsSumaries data table
@anvil.server.callable
def add_to_newsSummaries(pubDate, title, content, summary, topic_value, publication, source, link):
    # Check if story with same headline and author already exists
    existing_story = app_tables.newssummaries.get(headline=title, author=str(source))
    
    if existing_story:
        # Compare publication dates and update if the new story is more recent
        existing_pubDate = existing_story['pubDate']
        new_pubDate = datetime.strptime(pubDate, "%Y-%m-%d %H:%M:%S").date()
        if new_pubDate > existing_pubDate:
            existing_story.delete()
        else:
            return {"status": "skipped", "message": "Existing story is more recent or same date."}

    # Add the new story
    pubDate_datetime = datetime.strptime(pubDate, "%Y-%m-%d %H:%M:%S").date()
    app_tables.newssummaries.add_row(
        dateTimeAdded=datetime.now(),
        pubDate=pubDate_datetime, 
        headline=title,
        content=content,
        summary=summary,
        topic=topic_value,
        publication=publication,
        author=str(source),
        storyLink=link
    )

    return {"status": "added", "message": "Story successfully added to the database."}






##############################################
#Calls the WorldNews API and requests stories
#API reference is here https://worldnewsapi.com/
@anvil.server.callable
def get_risk_articles():
    def search_news(api_key, text, number=10, language='en', sort='publish-time', sort_direction='DESC'):
        url = "https://api.worldnewsapi.com/search-news"
        query = {
            'api-key': api_key,
            'text': text,
            'number': number,
            'language': language,
            'sort': sort,
            'sort-direction': sort_direction
        }
        response = requests.get(url, params=query)

        # Check if request was successful
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Request failed with status code {response.status_code}")
            return None

    api_key = anvil.secrets.get_secret('newsapi_key')  # Fetch API key from Anvil's Secret Service

    # Set your date constraint
    three_day_ago = datetime.now() - timedelta(days=3)

    # Call the API without the date constraint
    risk_news = search_news(api_key, "risk management" or "crisis management" or "crisis" or "cyber" or "compliance" or "governance", number=50)
    print(risk_news)

    risk_articles_list = []
    for news in risk_news['news']:
        # Get the article's publication date
        publish_date_str = news['publish_date']
        publish_date = datetime.strptime(publish_date_str, '%Y-%m-%d %H:%M:%S')

        # Check if the publication date falls within the last three days
        if publish_date >= three_day_ago:
            title = news['title']
            summary = news['text']
            link = news['url']
            source = news['source_country']  # Assuming this is the equivalent of 'rights'
            date = news['publish_date']  # The new API does not provide publish date

            risk_articles_list.append({
                'Headline': title,
                'Source': source,
                'Date': date,
                'Summary': summary,
                'Link': link,
            })

    return risk_articles_list


###############################################
#Calls the Newsdata.io API and requests stories
#API reference is here https://newsdata.io/

@anvil.server.background_task
def get_risk_articles_newsdata():
    def search_news(api_key, text, number=10, language='en', prioritydomain='top'): 
        url = "https://newsdata.io/api/1/news"
        categories = "business"
        query = {
            'apikey': api_key,
            'qInTitle': text,
            'category': categories,
            'prioritydomain': prioritydomain,
            'language': language
        }
        response = requests.get(url, params=query)

        if response.status_code != 200:
            return {"error": f"Request failed with status code {response.status_code}"}

        return response.json()

    api_key = anvil.secrets.get_secret('newsData_key')
    risk_news = search_news(api_key, "risk management OR crisis management OR crisis OR cyber OR compliance OR governance", number=50)

    if "error" in risk_news:
        return risk_news

    if 'results' not in risk_news:
        return []

    # Retrieve the topic from the 'topics' table once before the loop
    topic_row = app_tables.topics.get(topics='Risk and crisis')
    topic_value = topic_row if topic_row else 'Risk and crisis'

    risk_articles_list = []
    for news in risk_news['results']:
        title = news['title']
        content = news['content']
        summary = news['description'] 
        link = news['link']
        # Extract first creator or default to 'Unknown'
        source = news['creator'][0] if news['creator'] else 'Unknown'
        publication = news['source_id'] 
        pubDate = news['pubDate']

        risk_articles_list.append({
            'Headline': title,
            'Source': source,
            'Summary': summary,
            'Link': link,
            'pubDate': pubDate,
            'publication': publication
            #'content': content
        })

        anvil.server.call('add_to_newsSummaries', pubDate, title, content, summary, topic_value, publication, source, link)

    return risk_articles_list

###########################
##Calls the NewsCatcher API

@anvil.server.callable
def get_risk_articles_newscatcher():
  newsCatcher_API = anvil.secrets.get_secret('newsCatcher_API')  # Fetch API key from Anvil's Secret Service

  # Set your date constraint
  three_day_ago = datetime.now() - timedelta(days=3)
  
  # Set your query parameters
  querystring = {"q":"risk management OR crisis management OR crisis OR cyber OR compliance OR governance",
                 "lang":"en",
                 "from": three_day_ago.strftime('%Y-%m-%d'),
                 "page_size":50}

  headers = {"x-api-key": newsCatcher_API}

  response = requests.request("GET", 'https://api.newscatcherapi.com/v2/search', headers=headers, params=querystring)
  
  risk_news = response.json()
  return risk_news

  risk_articles_list = []
  for article in risk_news['articles']:
    title = article['title']
    if 'photo' not in title.lower():
        if article['topic'] in topics:
            date = article['published_date']
            source = article['rights']
            summary = article['summary']
            link = article['link']
            topic = article['topic']

            risk_articles_list.append({
                'Headline': title,
                'Source': source,
                'Date': date,
                'Summary': summary,
                'Link': link,
            })

    risk_articles_str = json.dumps(risk_articles_list, indent=4)
    print(f'*********************\nHeadline: {title}\nSource: {source}\nTopic: {topic}\nDate: {date}\nSummary: {summary}\nLink to article: {link}\n\n')

##############################################
# NewsLit API via Google Sheets

@anvil.server.callable
def get_risknews_newLit():
    # Open your Google Sheet
    db = app_files.newslitfeed
    ws = db["NewsLitFeed"]

    # Initialize HTML string
    html_string = "<html><body>"

    # Iterate over each row in the Google Sheet
    for row in ws.rows:
    # Skip if the row doesn't have expected structure
      if set(row.keys()) != {'publication_date', 'title', 'author', 'image_url', 'language', 'source', 'excerpt', 'url'}:
        continue
      publication_date = row['publication_date']
      if isinstance(publication_date, datetime):
        formatted_date = publication_date.strftime('%d %B %Y')
      elif isinstance(publication_date, (float, str)) and publication_date.replace('.', '', 1).isdigit():
        excel_date_value = float(publication_date)
        date_obj = to_datetime(excel_date_value, unit='D', origin='1899-12-30')
        formatted_date = date_obj.strftime('%d %B %Y')
      else:
        date_obj = datetime.strptime(publication_date, '%Y-%m-%d %H:%M:%S')
        formatted_date = date_obj.strftime('%d %B %Y')
      
        # Start story HTML
        story_html = f"""
        <h2>{row['title']}</h2>
        <p><b>{row['author']}</b> <i>{row['source']}</i>, {formatted_date}</p>
        <p>{row['excerpt']}</P>
        <p><a href='{row['url']}'>{row['source']}</a></p>
        <br><hr>
        
        """
        html_string += story_html
      
    html_string += "</body></html>"
    return html_string

##################################
@anvil.server.callable
def get_WEEKLY_risknews_newLit():
    # Open your Google Sheet
    db = app_files.newslitfeed
    ws = db["NewsLitArchiveCopy"]
    within_oneWeek = datetime.now() - timedelta(days=7)

    def string_to_datetime(date_string):
      return datetime.strptime(date_string, '%d %B %Y')

    # Initialize HTML string
    html_string = "<html><body>"

    # Iterate over each row in the Google Sheet
    for row in ws.rows:
    # Skip if the row doesn't have expected structure
      if set(row.keys()) != {'publication_date', 'title', 'author', 'image_url', 'language', 'source', 'excerpt', 'url'}:
        continue
      publication_date = row['publication_date']
      if isinstance(publication_date, datetime):
        formatted_date = publication_date.strftime('%d %B %Y')
      elif isinstance(publication_date, (float, str)) and publication_date.replace('.', '', 1).isdigit():
        excel_date_value = float(publication_date)
        date_obj = to_datetime(excel_date_value, unit='D', origin='1899-12-30')
        formatted_date = date_obj.strftime('%d %B %Y')
      else:
        date_obj = datetime.strptime(publication_date, '%Y-%m-%d %H:%M:%S')
        formatted_date = date_obj.strftime('%d %B %Y')
      
        # Start story HTML
        date_obj_for_comparison = string_to_datetime(formatted_date)
        if date_obj_for_comparison >= within_oneWeek:
          story_html = f"""
          <h2>{row['title']}</h2>
          <p><b>{row['author']}</b> <i>{row['source']}</i>, {formatted_date}</p>
          <p>{row['excerpt']}</P>
          <p><a href='{row['url']}'>{row['source']}</a></p>
          <br><hr>
        
          """
          html_string += story_html
      
    html_string += "</body></html>"
    html_string = remove_excessive_br_tags(html_string)
    return html_string


###############################
# Get news from the Brave API
@anvil.server.callable
def get_risk_articles_BraveAPI():
  brave_API = anvil.secrets.get_secret('brave_API')  # Fetch API key from Anvil's Secret Service
  
  # Set your query parameters
  querystring = {"q":"risk management",
                 "search_lang":"en",
                 "freshness": "pd",
                 #"result_filter": "news",
                 "count":20}

  headers = {"X-Subscription-Token": brave_API}
  response = requests.request("GET", 'https://api.search.brave.com/res/v1/web/search', headers=headers, params=querystring)

  
  risk_news = response.json()
  return risk_news


##############################################
#Splits the articles up so these don't exceed the token count for the model. Using 15K as a limit.
@anvil.server.callable
def split_articles(articles, max_tokens=15000):
    # This is a simple function to split the list of articles into smaller chunks that will fit within the token limit.

    split_articles = []
    current_chunk = []
    current_tokens = 0

    for article in articles:
        # Calculate the number of tokens in this article.
        # This is a rough estimate, you might need to adjust it.
        # Here, I'm assuming an average of 5 tokens per word.
        article_tokens = len(article['Headline'].split()) + len(article['Summary'].split()) * 5

        # If adding this article would exceed the token limit, start a new chunk.
        if current_tokens + article_tokens > max_tokens:
            split_articles.append(current_chunk)
            current_chunk = []
            current_tokens = 0

        # Add the article to the current chunk and update the token count.
        current_chunk.append(article)
        current_tokens += article_tokens

    # Add the last chunk if it's not empty.
    if current_chunk:
        split_articles.append(current_chunk)
    return split_articles

##############################################
#Sends the stories to ChatGPT to summarize and put into HTML format
@anvil.server.callable
def summarize_stories(risk_articles_str):
    openai.api_key = anvil.secrets.get_secret('openai_api')

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=[
          {"role": "system", "content": "You are an experienced risk and security analyst, responsible for writing succinct summary reports."},
          {"role": "user", "content": f"Please summarize these articles for a daily briefing email to help the reader understand each story. For each story, use the following format: Begin with an HTML <h1> tag for the headline. Follow this with the date in a new line. Then, summarize the story in a paragraph, ensuring the summary is concise yet informative. Conclude each story with a HTML <a> tag for the clickable source link where the link text is the source name. The articles should be separated by a horizontal line. Ignore articles that have no political, economic, security or geopolitical relevance. Return the results in HTML format to help formatting in Google Docs. Here are the articles: \n{risk_articles_str}"}
]

    )

    html_summary = completion.choices[0].message['content']
    return html_summary


##############################################
#Emails the news summaries
@anvil.server.callable()
def send_daily_news_email(html_summary):
    try:
        today = datetime.now()
        date = (today.strftime("%d-%B-%Y"))
        day = (today.strftime("%A"))

        # email header and footer
        header = f"""
            <html>
            <body>
            <h2>Happy {day}! Here's the risk news for {date}</h2>
        """

        footer = """
            <p>That's it for today. See you tomorrow</p>
            <p><i>~Andrew</i></p>
            </body>
            </html>
        """

        daily_subject = (f"Daily risk news for {date}")

        mail_message_body = header + html_summary + footer  # this is an HTML message

        # Send Email
        anvil.email.send(to='andrew@andrewsheves.com',
                         subject=daily_subject,
                         html=mail_message_body,
                         from_address='andrew@tarjumansolutions.com',
                         from_name='Andrew')

        return "Email sent successfully."
    except Exception as e:
        return f"Failed to send email: {e}"
  

@anvil.server.background_task
def send_daily_risk_summary():
    articles = get_risk_articles()  # Call the function directly
    split_articles_list = split_articles(articles)
    # Generate summaries
    summaries = []
    for chunk in split_articles_list:
        summary = summarize_stories(chunk)
        summaries.append(summary)
    # Join all summaries into one string
    full_summary = ''.join(summaries)

    # Now you can pass 'full_summary' to your 'send_daily_news_email' function
    send_daily_news_email(full_summary)

