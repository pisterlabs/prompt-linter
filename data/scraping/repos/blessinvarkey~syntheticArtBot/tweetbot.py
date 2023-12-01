from fileinput import filename
from os import access

from prompt_toolkit import prompt
import openai, tweepy, random, accessapi, time
client = tweepy.Client(consumer_key = accessapi.api_key, consumer_secret= accessapi.api_key_secret, access_token = accessapi.access_token, access_token_secret =accessapi.access_token_secret)
auth = tweepy.OAuthHandler(consumer_key = accessapi.api_key, consumer_secret= accessapi.api_key_secret, access_token = accessapi.access_token, access_token_secret =accessapi.access_token_secret)
openai.api_key = accessapi.openai_api_key

def bot(user_input):
    response = openai.Completion.create(
        engine = "text-davinci-001",
        prompt = user_input,
        max_tokens = 64
    )
    text = response.choices[0].text
    print(text)

    reply_y_or_n= input("Shall I print the tweet?")
    if reply_y_or_n == 'y':
        tweet = client.create_tweet(text = text)
        print("Tweet successful!")
    else:
        bot(prompt)

def replyBot():
    api = tweepy.API(auth)
    mentions = api.mentions_timeline()
    print(str(len(mentions)) + " number of statuses have been mentioned.")

    for mention in mentions:
        print("Mention Tweet Found")
        print(f"{mention.author.screen_name} - {mention.text}")
        print(mention.id)
        tweet_id =  mention.id

        response = openai.Completion.create(
            engine = "text-davinci-001",
            prompt = mention.text,
            max_tokens = 64
        )
        text = response.choices[0].text
        

        print(text)
        reply_y_or_n= input("Shall I print the tweet?")
        if reply_y_or_n == 'y':
            tweet = client.create_tweet(text = text)
            api.update_status('@'+mention.author.screen_name + text)

            print(tweet)
            print("Tweet successful!")
        else:
            bot()
#   time.sleep(15)
def postMedia(text, filename):
    api = tweepy.API(auth)
    media = api.media_upload(filename)
    print("The media ID is : " + media.media_id_string)
    print("The size of the file is : " + str(media.size) + " bytes")
    api.update_status(text, media_ids = [media.media_id_string])
    


if __name__ == "__main__":
#    prompt = "A joke on a "
#    replyBot()
    postMedia('Pink Elephant, Gond art #dallemini #craiyon #gondart' , 'images/gond_elephant.png')
#    bot(prompt)
