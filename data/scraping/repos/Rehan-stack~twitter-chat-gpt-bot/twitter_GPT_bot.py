
import tweepy
import openai
import time

consumer_key = ''
consumer_secret = ''
access_token = ''
access_token_secret = ''

# Insert your OpenAI API key here
openai_key = ""

auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)

# Authenticate with OpenAI API
openai.api_key = openai_key

# Set the starting ID for mentions
replied_tweets = []

while True:
    # Search for tweets that mention the account
    mentions = api.mentions_timeline(count=100)

    if len(mentions) == 0:
        time.sleep(20)
        continue

    # Iterate over the mentions
    for mention in mentions:
        # Check if the tweet has already been replied to
        if mention.id not in replied_tweets:
            # Use GPT-3 to generate a reply
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=f"@{mention.user.screen_name} {mention.text[:280]}",
                max_tokens=250,
                temperature=0.5,
            )

            print(response.choices[0].text)

            # Post the reply
            api.update_status(
                f"@{mention.user.screen_name} {response.choices[0].text[:280]}",
                in_reply_to_status_id=mention.id,
            )
            print(f"{mention.id} reply tweet done")

            # Add the tweet to the list of replied tweets
            replied_tweets.append(mention.id)

    time.sleep(20)
