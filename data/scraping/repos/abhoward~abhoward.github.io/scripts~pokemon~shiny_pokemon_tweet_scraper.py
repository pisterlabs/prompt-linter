import pandas as pd
import re
import json

import tweepy
import openai

import os
from dotenv import load_dotenv

load_dotenv()
TWEEPY_ACCESS_TOKEN = os.getenv('TWEEPY_ACCESS_TOKEN')
OPENAI_SECRET = os.getenv('OPENAI_SECRET')
OPENAI_MODEL = os.getenv('OPENAI_MODEL')

data_file_path = "..\\..\\data\\pokemon\\{}"

auth = tweepy.OAuth2BearerHandler(TWEEPY_ACCESS_TOKEN)
api = tweepy.API(auth)
client = tweepy.Client(TWEEPY_ACCESS_TOKEN)
openai.api_key = OPENAI_SECRET

with open(data_file_path.format('safari_week_2023_tweets.json')) as f:
    safari_week_2023_tweets = json.load(f)
    
with open(data_file_path.format('safari_week_2023_gpt_responses.json')) as f:
    gpt_responses = json.load(f)

query = '(#safariweek2023 OR #safariweek) -is:retweet' # change this if you want to get different tweets
counter = 0
last_tweet_id = max(safari_week_2023_tweets.keys())

for tweet in tweepy.Paginator(client.search_recent_tweets, query, since_id = last_tweet_id, max_results = 100).flatten():
    if tweet.id != 1667931338324996096: # this is a duplicate tweet 
        safari_week_2023_tweets[str(tweet.id)] = tweet.text
        counter = counter + 1

print('Number of new tweets pulled: {}'.format(counter))

with open(data_file_path.format('safari_week_2023_tweets.json'), 'w') as f:
    json.dump(safari_week_2023_tweets, f)

cleaned_tweets = {}

for tweet_id, text in safari_week_2023_tweets.items():
    text = re.sub(r" http\S+", "", text) # remove link
    text = re.sub(r"#\S+", "", text) # remove all hashtags
    text = text.strip()
    if text != '':
        cleaned_tweets[str(tweet_id)] = text

pokemon_text_file = open(data_file_path.format('pokemon_list.txt'), 'r')
mon_names = pokemon_text_file.read().split('\n')

new_tweets = {}

for tweet_id, tweet in cleaned_tweets.items():
    if tweet_id not in list(gpt_responses.keys()):
        new_tweets[tweet_id] = tweet

model_engine = OPENAI_MODEL

for tweet_id, tweet in new_tweets.items():
    try:
        response = openai.Completion.create(model = model_engine, prompt = tweet + '\nn###\n\n', stop = ' ++++')
        gpt_responses[str(tweet_id)] = response.choices[0].text.strip()
    except openai.error.APIError as e:
        print(f"OpenAI API returned an API Error: {e}")
        pass

with open(data_file_path.format('safari_week_2023_gpt_responses.json'), "w") as f:
    json.dump(gpt_responses, f)

shiny_mons = []
shiny_encounters = []
shiny_tweets = []

for i, j in gpt_responses.items():
    if j.lower() != 'no pokemon found':
        if ' and ' in j:
            try:
                parse = j.split(' and ')
                for word in parse:
                    mon = word.split(': ')[0]
                    encounter = word.split(': ')[1]
                    if (mon not in mon_names) and (mon != 'No Pokemon found'):
                        print('Unknown Pokemon found: {} in tweet: {}'.format(mon, i))
                    else:
                        shiny_tweets.append(i)
                        shiny_mons.append(mon)
                        shiny_encounters.append(int(encounter))
            except:
                print('Error with tweet: {}'.format(i))
        else:
            try:
                mon = j.split(': ')[0]
                encounter = j.split(': ')[1]
                if (mon not in mon_names) and (mon != 'No Pokemon found'):
                    print('Unknown Pokemon found: {} in tweet: {}'.format(mon, i))
                else:
                    shiny_tweets.append(i)
                    shiny_mons.append(mon)
                    try:
                        shiny_encounters.append(int(encounter))
                    except ValueError:
                        print('Unable to int: {}').format(encounter)
            except:
                print('Error with tweet: {}'.format(i))

# check to make sure everything lines up
print('Shiny Tweets: {}\n\nShiny Pokemon: {}\n\nShiny Encounters: {}'.format(len(shiny_tweets), len(shiny_mons), len(shiny_encounters)))

shinies_df = pd.DataFrame({'tweet_id': shiny_tweets, 'pokemon': shiny_mons, 'encounters': shiny_encounters})
shinies_df['pokemon'] = shinies_df['pokemon'].replace('No Pokemon found', 'Unknown').replace('Mr. Mime', 'Mr Mime') # the period in Mr. Mime causes issues when loading the data labels in Highcharts
shinies_df['tweet_link'] = 'https://twitter.com/twitter/status/' + shinies_df['tweet_id'].apply(str)

mon_encounters_df = shinies_df[shinies_df['encounters'] > 0][['pokemon', 'encounters']].reset_index(drop = True).rename(columns={"pokemon": "name", "encounters": "y"})

encounters_dict = {}
encounters_dict['data'] = mon_encounters_df['y'].tolist()

mon_counts_df = shinies_df['pokemon'].value_counts().reset_index()
mon_counts_df.rename(columns = {'pokemon': 'name', 'count': 'y'}, inplace = True)
mon_counts_df['drilldown'] = mon_counts_df['name'] + ' Encounters'

drilldown = []
for mon in shinies_df['pokemon'].unique():
    drilldown.append(
        {
            'name': mon,
            'id': mon + ' Encounters',
            'data': shinies_df[shinies_df['pokemon'] == mon][['tweet_link', 'encounters']].values.tolist()
        }
    )

with open(data_file_path.format('safari_week_2023_mon_encounters.json'), 'w') as f:
    json.dump(drilldown, f)
    
with open(data_file_path.format('safari_week_2023_encounters.json'), 'w') as f:
    json.dump(encounters_dict, f)
    
mon_counts_df.to_json(data_file_path.format('safari_week_2023_mon_counts.json'), orient = 'records')
shinies_df.to_json(data_file_path.format('safari_week_2023_data.json'), orient = 'records')