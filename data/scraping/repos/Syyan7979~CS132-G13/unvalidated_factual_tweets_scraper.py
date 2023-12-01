import openai
import snscrape.modules.twitter as sntwitter
from dotenv import dotenv_values
import pytz
import pandas as pd

# Load environment variables
env_vars = dotenv_values('.env')

# Set OpenAI API key
openai.api_key = env_vars['OPEN_AI_KEY']
openai.organization = env_vars['OPEN_AI_ORG']

# Set the query string
query = "President Noynoy Aquino selling Scarborough shoal is still unconfirmed"

tweets = []
max_results = 150

counter = 150

for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    # Break the loop if the desired number of tweets is reached
    if len(tweets) >= max_results:
        break
    else:
        tweet_types = []
        # identify the type of tweet
        if tweet.rawContent:
            tweet_types.append("Text")
        if tweet.media:
            for media in tweet.media:
                if isinstance(media, sntwitter.Photo):
                    tweet_types.append("Image")
                elif isinstance(media, sntwitter.Video):
                    tweet_types.append("Video")
        if tweet.links:
            tweet_types.append("URL")
        if tweet.inReplyToTweetId or tweet.quotedTweet:
            tweet_types.append("Reply")

        tweet_types = list(set(tweet_types))

        counter += 1
        tweets.append(
            [
                    f"13-{counter}",
                    tweet.url,
                    f"@{tweet.user.username}",
                    tweet.user.displayname,
                    tweet.user.rawDescription,
                    "",
                    tweet.rawContent,
                    ", ".join(tweet_types),
                    tweet.date.astimezone(pytz.timezone('Asia/Manila')).strftime("%Y-%m-%d %H:%M:%S"),
                    tweet.likeCount,
                    tweet.replyCount,
                    tweet.retweetCount,
                    tweet.quoteCount,
            ]
        )

# Create a dataframe from the tweets list above
df = pd.DataFrame(
    tweets,
    columns=[
        "ID",
        "Tweet URL",
        "Account handle",
        "Account name",
        "Account bio",
        "Account type",
        "Tweet",
        "Tweet Type",
        "Date posted",
        "Likes",
        "Replies",
        "Retweets",
        "Quote Tweets"
    ]
)

# Save the dataframe as a CSV file
df.to_csv("unvalidated_factual_tweets.csv", index=False)