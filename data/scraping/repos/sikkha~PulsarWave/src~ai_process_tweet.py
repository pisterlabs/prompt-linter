###############################################################################
# Proces tweet by AI to display on Trend Radar
# Written by Kan Yuenyong (kan.yuenyong@siaintelligenceunit.com)
# Version 1.11
# Sat Jun 10 02:46:14 UTC 2023
#
# - Bug Fixed (Change retrieve tweeter id instead of tweeter account)
#
#

import openai
import os
import tweepy
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain import PromptTemplate
from langchain.llms import OpenAI

# Set Twitter API
# api_key = "YOUR_TWITTER_API_KEY"
# api_secrets = "YOUR_TWITTER_API_SECRETS"
# bearer_token = "YOUR_TWITTER_BEARER_TOKEN"
# access_token = "YOUR_TWITTER_ACESS_TOKEN"
# access_secret = "YOUR_TWITTER_ACCES_SECRET"

api_key = os.environ["API_KEY"]
api_secrets = os.environ["API_SECRETS"]
bearer_token = os.environ["BEARER_TOKEN"]
access_token = os.environ["ACCESS_TOKEN"]
access_secret = os.environ["ACCESS_SECRET"]


#import time

auth = tweepy.OAuth1UserHandler(api_key, api_secrets, access_token, access_secret)
api = tweepy.API(auth)

# Assuming results is a list
results = []

twitter_usernames = ['WIRED', 'ForeignPolicy', 'politico', 'CFR_org', 'business', 'TheEconomist', 'TheAtlantic', 'sciencenews', 'guardianeco', 'WebMD', 'insidehighered', 'ESPN', 'Variety']

for username in twitter_usernames:
    user = api.get_user(screen_name=username)
    user_id = user.id_str
    print("Printing latest 2 tweets from " + username + " (ID: " + user_id + ")")

    print("="*50)

    # Get the latest 2 tweets from the user's timeline using their ID
    latest_tweets = api.user_timeline(user_id=user_id, count=2)

    for i, tweet in enumerate(latest_tweets):
        print("Tweet {}: {}".format(i+1, tweet.text))
        
        # Add summaries to the dictionary
        results.append(tweet.text)


    print("\n")
    #time.sleep(15)  # Sleep for 15 seconds


#----------- PART1: ASK GPT to select tweet ------
#
# Initialize the OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

template = """
Proceed the command based on the context below. If the
command cannot be proceeded the information provided answer
with "I don't know".

Context: {context}

Question: {query}

Answer: """


# Define the context variable
context = """
From the following tweet, select the top 4 most important tweets, 
and 1 tweet that is not that important but imply the long term consequence that we can't ignore. 
Print only the content and related link in tweet you've selected. 
No need to identify the source of tweet where they are from. 
Don't put quotation marks "..." on the output. Don't put the numbering in front of each tweet you've selected. 
The last tweet don't put the line "One Tweet with Long-Term Consequences:".
"""

# Read query from a file
#with open('/tmp/jj.txt', 'r') as file:
#    query = file.read().split("\n")

query = results
    
# Create a PromptTemplate object
prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template=template
)

# Create a new essay by formatting the context and query parameters with the PromptTemplate object
newessay = prompt_template.format(context=context, query=query)

# Initialize an OpenAI object
openai_instance = OpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=openai.api_key
)

# Generate a response from the OpenAI API
response = openai_instance(newessay)

# Print the response to the console
#print(response)

# Print the response to safe into file
with open('/tmp/select_tweet.txt', 'w') as file:
    file.write(str(response))

    
    #----------- PART1: ASK GPT to select tweet ------
#
# Initialize the OpenAI API key
openai.api_key = os.environ["OPENAI_API_KEY"]

template = """Answer the question based on the context below. If the
question cannot be answered using the information provided answer
with "I don't know".

Context: {context}

Question: {query}

Answer: """


# Define the context variable
context = """
From the following tweet, select the top 4 most important tweets, 
and 1 tweet that is not that important but imply the long term consequence that we can't ignore. 
Print only the content and related link in tweet you've selected. 
No need to identify the source of tweet where they are from. 
Don't put quotation marks "..." on the output. Don't put the numbering in front of each tweet you've selected. 
The last tweet don't put the line "One Tweet with Long-Term Consequences:".
"""

# Read query from a file
#with open('/tmp/jj.txt', 'r') as file:
#    query = file.read().split("\n")

query = results
    
# Create a PromptTemplate object
prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template=template
)

# Create a new essay by formatting the context and query parameters with the PromptTemplate object
newessay = prompt_template.format(context=context, query=query)

# Initialize an OpenAI object
openai_instance = OpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=openai.api_key
)

# Generate a response from the OpenAI API
response = openai_instance(newessay)

# Print the response to the console
#print(response)

# Print the response to safe into file
with open('/tmp/select_tweet.txt', 'w') as file:
    file.write(str(response))



#----------- PART2: ASK GPT to process tweet ------
#


context = """
process the following command, print final output in section 2 for each output from the process on a separate line, the new output will be on the new line, don't put any explanation on it:


1) definition section:

do categorize the following tweet quadrant 0 is "societal and cultural", 1 is "innovation and law", 2 is "politics and security", 3 is "economics"

ring 0 is prioritized the most important as "engage", 1 is "monitor", 2 as "observe" and 3 as "acknowledge" (or lowest priority)

label is to summarize the text in the tweet in merely not more than 4 words

link is to extract url in the field

active and move give them True and 0

example of the tuple will be like (make it in MD format), there should be no blank line between each line of output:

1, 1, Docker, https://www.docker.com/, True, 0\n
0, 2, Cyber Sovereignty Pursued, https://t.co/BMm7PxWZ7o, True, 0\n
1, 2, Ethernet Connection Guide, https://t.co/xcNmh8Kt1M, True, 0

2) command section:

now process the following tweets:
"""


query = response
    
# Create a PromptTemplate object
prompt_template = PromptTemplate(
    input_variables=["context", "query"],
    template=template
)


# Create a new essay by formatting the context and query parameters with the PromptTemplate object
newessay = prompt_template.format(context=context, query=query)

# Initialize an OpenAI object
openai_instance = OpenAI(
    model_name="gpt-4",
    openai_api_key=openai.api_key
)

# Maximum attempts to generate a response
max_attempts = 3

for attempt in range(max_attempts):
    # Generate a response from the OpenAI API
    response2 = openai_instance(newessay)

    # If the response doesn't contain "I don't know.", break the loop
    if "I don't know." not in response2:
        break

# Add new line to the operated string
response2 += "\n"

# Print the response to safe into file
with open('/tmp/process_tweet.txt', 'w') as file:
    file.write(str(response2))
