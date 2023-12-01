import openai
import schedule
import time
import tweepy

# Setting up the API keys
client = tweepy.Client(
    consumer_key="Your Consumer Key",
    consumer_secret="Your Consumer Key Secret",
    access_token= "Your Access Token",
    access_token_secret="Your Access Token Secret",
    )

# Setting up OAuth for authorization
auth = tweepy.OAuthHandler(client.consumer_key, client.consumer_secret)
auth.set_access_token( client.access_token ,client.access_token_secret)
api = tweepy.API(auth,wait_on_rate_limit= True )

# Setting up OpenAI API
openai.api_key = "Your  openai Api Key"

def generate_tweet():
    prompt = " Your prompt describing the type of tweets you want "
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt= prompt,
        max_tokens=50,
        temperature=0.9,
    )
    tweet = response["choices"][0]["text"]
    return tweet


# Posting the tweet on the wall
def post_tweet():
    tweet = generate_tweet()
    client.create_tweet(text=tweet)
    #api.update_status(tweet)
    print("Tweet posted:", tweet)

# Scheduling to tweet every day
schedule.every().day.at("12:00").do(post_tweet)

while True:
    schedule.run_pending()
    time.sleep(1)

