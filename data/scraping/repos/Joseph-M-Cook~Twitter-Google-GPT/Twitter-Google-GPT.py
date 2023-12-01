import openai
import requests
from requests_oauthlib import OAuth1
import json
import gradio as gr
from googleapiclient.discovery import build
from datetime import date
import calendar

# Twitter API credentials
consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""

# OpenAI API key
openai.api_key = ""

# Google Custom Search API credentials
GOOGLE_DEV_KEY = ""
GOOGLE_CX_KEY = ""

# Function to generate a Twitter search query using OpenAI's GPT-4
def get_twitter_search_query(query):
    messages = [{"role": "system", "content": 
                 "You are an AI assistant that helps to convert text into a relevant Twitter search API query.\n"
                 "You output only 1 query for the latest message and nothing else.\n"

                 "Info:\n"
                 'Operator: keyword	Type: Standalone Example: pepsi OR cola OR "coca cola"\n'

                 'Examples:\n'
                 'Which NHL games are on tonight?: ("nhl news" OR "nhl tonight" OR "hockey games" OR "hockey tonight") -is:retweet lang:en -has:links -is:reply\n'
                 'What is some recent soccer news?: ("soccer news" OR "football news" OR "soccer updates" OR "football updates") -is:retweet -is:reply lang:en -has:links -is:reply\n'
                 'What stocks are people buying?: ("stocks" OR "stock market" OR "investing" OR "investments") ("buying" OR "purchasing" OR "investing") -is:retweet -is:reply lang:en -has:links\n'}]

    messages.append({"role": "user", "content": 
                     "Based on my previous messages,\n"
                     "What is the most relevant Twitter search query for the text below?\n\n"
                     "Text: " + query + "\n\n"
                     "Query:"})
    
    search_query = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0,
    )['choices'][0]['message']['content']

    print(search_query.strip("\""))
    return search_query.strip("\"")
    
# Function to execute a Twitter search using the given query and return the tweets found
def twitter_search(query):
    search_url = "https://api.twitter.com/2/tweets/search/recent"

    query_params = {
        'query': query,
        'tweet.fields': 'author_id',
        'user.fields': 'username',
        'expansions': 'author_id',
        'max_results': 50
    }

    auth = OAuth1(consumer_key, consumer_secret, access_token, access_token_secret)

    # Function to connect to Twitter API endpoint and return the JSON response
    def connect_to_endpoint(url, params):
        response = requests.get(url, auth=auth, params=params)
        if response.status_code != 200:
            raise Exception(response.status_code, response.text)
        return response.json()

    # Function to parse the JSON response and return the tweets as a string
    def print_tweets(json_response):
        i = 0
        all_tweets = ""
        if 'data' in json_response:
            for tweet in json_response['data']:
                user = next(user for user in json_response['includes']['users'] if user['id'] == tweet['author_id'])
                tweet_url = f"https://twitter.com/{user['username']}/status/{tweet['id']}"
                tweet_text = f"{user['username']}: {tweet['text']}\n"
                all_tweets += tweet_text
        return all_tweets

    json_response = connect_to_endpoint(search_url, query_params)

    all_tweets = print_tweets(json_response)
    
    return all_tweets

# Function to generate AI response to orignal question based on fetched tweets
def Twitter_AIResponse(query, tweets):
        messages = [{"role": "system","content": 
                     "You are a bot that answers questions to the best of your ability based on search results from twitter."
                     "Do not apologize or mention what you are not capable of."
                     "do not start your response with anything like 'Based on the search results'"}]
    
        messages.append({"role": "user", "content": 
                         "Answer the question to the best of your ability based on the search results and the query"
                         "results: " + tweets + "\n\n"
                         "Query:" + query})

        search_query = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0,
        )['choices'][0]['message']['content']

        return search_query


# Reference: https://github.com/VRSEN/chatgtp-bing-clone
class Google_Search():
    # Function to initalize Google Custom Search API
    def __init__(self):
        self.service = build("customsearch", "v1", developerKey=GOOGLE_DEV_KEY)

    # Function to execute Google Search API with generated query
    def _search(self, query):
        response = (self.service.cse().list(q=query,cx=GOOGLE_CX_KEY,).execute())
        return response['items']

    # Function to construct Google query
    def _get_search_query(self, query):
        messages = [{"role": "system","content": 
                     "You are an assistant that helps to convert text into a web search engine query."
                     "You output only 1 query for the latest message and nothing else."}]

        messages.append({"role": "user", "content": 
                         "Based on my previous messages,\n"
                         "what is the most relevant and general web search query for the text below?\n\n"
                        f"For context (if nessecary) it is: {date.today().strftime('%B')} {date.today().strftime('%d')} {date.today().strftime('%Y')}\n"
                        #"For context (if nessecary) it is: mid may 2023"
                         "Text: " + query + "\n\n"
                         "Query:"})

        search_query = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0,
        )['choices'][0]['message']['content']

        return search_query.strip("\"")

    # Function to construct response
    def run_text(self, query):
        search_query = self._get_search_query(query)

        messages = [{"role": "system","content": 
                     "You are a financial assistaint that extracts all relevant data based on search results and "
                     "provides links at the end to relevant parts of your answer. Keep your summaries very brief"
                     "Do not apologize or mention what you are not capable of"}]

        prompt = "You are a financial assistaint, extract all relevant information from the search results below: \n\n"
        results = self._search(search_query)

        for result in results:
            prompt += "Link: " + result['link'] + "\n"
            prompt += "Title: " + result['title'] + "\n"
            prompt += "Content: " + result['snippet'] + "\n\n"
        prompt += "\nQuery: " + query

        messages.append({"role": "user", "content": prompt})

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.2,
        )['choices'][0]['message']['content']

        return response
    
# Function to get final response using Twitter and Google Results
def AIResponse(query, tweets, google):
        messages = [{"role": "system", "content": 
                     "You are a bot that answers questions to the best of your ability based on search results from twitter and a google search.\n"
                     "Do not apologize or mention what you are not capable of." 
                     "Use line breaks to split your response into 1-3 paragraphs."
                     "Do not say anything like 'Google search results show'"}]
    
        messages.append({"role": "user", "content": 
                         "Answer the question step by step to the best of your ability based on the search results and the query\n" 
                         "Twitter results: " + tweets + "\n\n"
                         "Google Results: " + google + "\n"
                         "Query:" + query})
        print(messages)
        Final_Answer = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            temperature=0.2,
        )['choices'][0]['message']['content']

        return Final_Answer

# Main function
def main(query_text):
    Google = Google_Search()
    data = Google.run_text(query_text)


    print(data)

    generated_query = get_twitter_search_query(query_text)
    ans = twitter_search(generated_query)
    Twitter_Answer = Twitter_AIResponse(query_text, ans)
    return AIResponse(query_text, Twitter_Answer, data)


# Interface and Execution
interface = gr.Interface(
    fn=main,
    inputs=[gr.inputs.Textbox(lines=3, label="Question:")],
    outputs=[gr.outputs.Textbox(label="Output:")],
    title="Twitter-Google-GPT",
    description="Twitter-Google-GPT is an AI tool that utilizes OpenAI's GPT-4 to transform your questions into search queries for Twitter and Google, yielding concise, relevant responses from diverse sources.",
)
interface.launch()
