import openai
import pandas as pd
import numpy as np
import tweepy
import webbrowser
import re

'''
Uses OpenAI's GPT-3 to classify sentiment of tweets

1. Dev docs: https://beta.openai.com/docs/introduction
2. Dataset: https://www.kaggle.com/gpreda/all-covid19-vaccines-tweets
3. Strip out Countries: https://github.com/grammakov/USA-cities-and-states

TODO's:
1. Preprocessing
    - Get rid of newlines, link to tweet
    - Link tweets to a specific vaccine: string match by vaccine name
    (pfizer, moderna, j&j, AZ) and use one-hot encoding basically every
    vaccine has a column in the DF and then 0 if not or 1 if present

2. Model Run
    - Run

3. Postprocessing
    - Output .csv will have sentiments mapped to each but need to decide
    what to visualize
'''

openai.api_key = open('./GPT_SECRET_KEY.txt', 'r').readline().rstrip()

df = pd.read_csv('./vaccination_all_tweets.csv')

# strip out non-us tweets
cities_file = open('./us_cities_states_counties.csv', 'r')
states_list = set()

for line in cities_file:
    abrev = "".join(line.split("|")[1:2]).lower()

    if len(abrev) == 2:
        states_list.add(abrev)

states_list = list(states_list)

to_drop = []
for idx, row in df.iterrows():
    try:
        curr = row["user_location"].lower().split(" ")[-1]
        if curr not in states_list and curr[:3] != "usa" and curr != "america":
            to_drop.append(idx)

    except:
        to_drop.append(idx)

print(len(df) - len(to_drop))
df = df.drop(to_drop)
print(len(df))

df["pfizer"] = 0
df["moderna"] = 0
df["j&j"] = 0
df["AZ"] = 0

to_drop = []

for idx, row in df.iterrows():

        lower_tweet = row["text"].lower()
        no_newline = lower_tweet.replace('\n',"")
        clean_tweet = re.sub(r"http\S+", "", no_newline)
        # print(clean_tweet)
        df.at[idx,"text"] = clean_tweet

        # assign tweet to certain type of vaccine
        if "pfizer" in clean_tweet:
            df.at[idx, "pfizer"] = 1
        if "moderna" in clean_tweet:
            df.at[idx, "moderna"] = 1
        if "j&j" in clean_tweet or "johnson & johnson" in clean_tweet:
            df.at[idx, "j&j"] = 1
        if "az" in clean_tweet or "astrazeneca" in clean_tweet:
            df.at[idx, "AZ"] = 1

ctr = 0
for idx, row in df.iterrows():
    curr = row["text"].split("https:")
    if len(curr) == 2:
        ctr += 1
print(ctr)

get tweet string from link if the whole string is not included
twitter_consumer_key = open('./TWITTER_KEY.txt', 'r').readline().rstrip()
twitter_consumer_secret = open('./TWITTER_SECRET_KEY.txt', 'r').readline().rstrip()

callback_uri = 'oob'
auth = tweepy.OAuthHandler(twitter_consumer_key, twitter_consumer_secret, callback_uri)
redirect_url = auth.get_authorization_url()
webbrowser.open(redirect_url)

user_pint_input = input("Pin? ")
auth.get_access_token(user_pint_input)
api = tweepy.API(auth, wait_on_rate_limit=True, retry_count=5, retry_delay=30)

for tweet in tweepy.Cursor(api.search,q="#PfizerBioNTech",count=100, lang="en", since="2020-01-01", tweet_mode="extended").items():
    print (tweet.created_at, tweet.full_text)

to_drop = []

for idx, row in df.iterrows():
    curr = row["id"]
    try:
        print(curr)
        tweet = tweepy.Cursor(api.search, id=curr, tweet_mode="extended")
        # tweet = api.get_status(curr, tweet_mode="extended")
        print(tweet.full_text)
        # print(tweet.full_text.split("http")[1].repace('\n',""))
        # cleanup/link to vax
        df.at[idx,"text"] = tweet.full_text
    except:
        print("MISSED")
        to_drop.append(idx)

# print(len(df) - len(to_drop))
df = df.drop(to_drop)
# print(len(df))

# df.to_csv('./data.csv')


df["sentiment"] = ""
df["clssification"] = ""

# remove .head() when we run for real...
for idx, row in df.iterrows():
    if row["text"] != np.nan:
        data = " ".join(row["text"].split(" ")[:-1])
        sentiment_json = openai.Classification.create(
            model="curie",
            query=data,
            labels=["Positive", "Negative", "Neutral"],
            examples=[["Happy and relieved to have the #PfizerBioNTech #Covidvaccine – amazing work from all at @MHRAgovuk and @NHSuk since it’s less than 2 weeks since CHM recommended that it be approved in the UK. #GetVaccinated!", "Positive"],
            ["The trump administration failed to deliver on vaccine promises, *shocker* #COVIDIOTS #coronavirus #CovidVaccine", "Negative"],
            ["Will you be taking the COVID-19 vaccine once available to you? #COVID19 #Pfizer #BioNTech #vaccine #PfizerBioNTech", "Neutral"]]
        )
        df.at[idx,"sentiment"] = sentiment_json["label"]
        classification_json = openai.Classification.create(
            model="curie",
            query=data,
            labels=["Hopeful", "Fearful", "Neutral"],
            examples=[["Finally, #vaccine started in all the EU. #vaccineday is truly historic bringing hope and relief to millions of people. Thanks to #EU for supporting it and buying it for us. 🙏 to #PfizerBioNTech #AstraZeneca #Moderna. Now, most people should do it for everyone to be safe. I will.", "Hopeful"],
            ["And this is the state of the poor quality #Sinovac. Even friendly countries are scared about it", "Fearful"],
            ["Will you be taking the COVID-19 vaccine once available to you? #COVID19 #Pfizer #BioNTech #vaccine #PfizerBioNTech", "Neutral"]]
        )
        df.at[idx,"classification"] = classification_json["label"]
print(df["sentiment"].head())
print(df["classification"].head())
