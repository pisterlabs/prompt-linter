import tweepy
import json
import openai
import urllib.request
import io

bearer_token = "YOUR BEARER TOKEN"
access_token = "YOUR ACCESS TOKEN"
access_token_secret = "YOUR ACCESS TOKEN SECRET"
consumer_key = "YOUR CONSUMER KEY"
consumer_secret = "YOUR CONSUMER SECRET"
open_ai_key = "YOUR OPEN AI KEY"

# Authenticate to Twitter
client = tweepy.Client(consumer_key=consumer_key, consumer_secret=consumer_secret,
    access_token=access_token, access_token_secret=access_token_secret)

auth =  tweepy.OAuth1UserHandler (consumer_key, consumer_secret, access_token, access_token_secret)
api = tweepy.API(auth)

# Authenticate to OpenAI
openai.api_key = open_ai_key


# Create stream rule for twitter
handle = "YOUR HANDLE"
bot_twitter_id = "YOUR BOTS TWITTER ACCOUNT ID"
mentionsRule = tweepy.StreamRule(handle + " is:reply has:mentions -from:" + bot_twitter_id + " -to:" + bot_twitter_id)

# Call Open AI to generate image
def illustrate_tweet(tweet_text):
    print(tweet_text)
    response = openai.Image.create(
        prompt = tweet_text,
        n = 1,
        size = '512x512'
    )

    image_url = response['data'][0]['url']
    return image_url

# Create tweet to post
def reply_to_tweet(og_tweet, reply_id):
    image_url = illustrate_tweet(og_tweet.text)
    media_id = upload_image(image_url)
    client.create_tweet(
        text="Made this original piece for you:",
        media_ids= [media_id],
        in_reply_to_tweet_id=reply_id,
        user_auth=True
    )

# Get information of the top tweet in the thread where the bot was mentioned
def get_original_tweet(id):
    tweet_reply = client.get_tweet(id, tweet_fields=["conversation_id"], user_auth=True)
    tweet_to_illustrate_id = tweet_reply.data.conversation_id
    tweet_to_illustrate_response = client.get_tweet(tweet_to_illustrate_id,user_auth=True)
    tweet_to_illustrate =  tweet_to_illustrate_response.data
    return (tweet_to_illustrate)

# Convert response to JSON
def process_response(data):
    d = json.loads(data)
    return d

# Upload image to Twitter
def upload_image(image_url):
    data = urllib.request.urlopen(image_url).read()
    file_like_object = io.BytesIO(data)
    media = api.simple_upload("TweetArt.png",file=file_like_object)
    print(media.media_id)
    return media.media_id

# Override of Stream client object to modify on_data method
class MentionsPrinter (tweepy.StreamingClient):
    def on_connect(self):
        print("Connected to the streaming API.")

    def on_data(self, data):
        d = process_response(data)
        tweet_id = d['data']['id']
        #og_tweet has properties text and id
        og_tweet = get_original_tweet(tweet_id)
        reply_to_tweet(og_tweet, tweet_id)

# Create stream client and start filtering
mentionsPrinter = MentionsPrinter(bearer_token)
mentionsPrinter.filter()
